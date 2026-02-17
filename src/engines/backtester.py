import pandas as pd
import numpy as np

from src.utils.signal_processing import apply_sticky_breakout_filter


class BacktestEngine:
    """
    BacktestEngine executes trade simulations using SDE regime signals.
    Features SNR-based dynamic scaling, ADX-based hold boosting, 
    and toggleable filters for research flexibility.
    """
    def __init__(self, config_loader):
        """
        Initializes the engine with settings from the configuration loader.
        """
        self.configuration = config_loader.get_backtest_settings()
        self.trading_parameters = self.configuration['trading_parameters']
        self.risk_parameters = self.configuration['risk_management']
        self.scaling_parameters = self.configuration['parameter_scaling']
        self.execution_costs = self.configuration['execution_costs']

        self.filter_settings = self.configuration.get('filters', {
            'use_sticky': True,
            'use_adx': True,
            'only_selected_zone': False
        })

    def generate_regime_signals(
        self,
        price_data: pd.DataFrame,
        k_estimated: float,
        gamma_estimated: float,
        ):
        """
        Calculates regime probabilities and generates sticky breakout signals.
        """
        entry_threshold = self.risk_parameters.get('entry_probability_threshold', 0.5)
        minimum_duration = self.risk_parameters.get('minimum_signal_duration', 5)

        # Calculate Sigmoid-based Regime Probability
        price_data['regime_prob'] = 1 / (1 + np.exp(-k_estimated * (price_data['hybrid_z_score'] - gamma_estimated)))

        # Apply Sticky Filter (Persistence check)
        binary_entries = (price_data['regime_prob'] > entry_threshold).astype(int)
        price_data['sticky_signal'] = apply_sticky_breakout_filter(binary_entries,
                                                                   minimum_duration=minimum_duration)

        return price_data

    def adjust_parameters_by_snr(self, estimated_params: dict, params_base: dict):
        """
        Scales TP, SL, and holding duration based on SNR and estimated volatility.
        """
        reference_sigma = self.risk_parameters.get('reference_sigma_1', 14.665)
        sigma_1_estimated = estimated_params.get('sigma_1', reference_sigma)
        vol_quality = sigma_1_estimated / reference_sigma

        snr_long = estimated_params.get('alpha_long', 30.0) / sigma_1_estimated
        snr_short = estimated_params.get('alpha_short', 20.0) / sigma_1_estimated

        scale = self.scaling_parameters
        snr_div = scale['snr_divisor']

        return {
            'tp_long': params_base['tp_long'] * np.clip(snr_long / snr_div, *scale['tp_long_clip']),
            'sl_long': params_base['sl_long'] * vol_quality * np.clip(1.0 / (snr_long / snr_div), *scale['sl_long_clip']),
            'tp_short': params_base['tp_short'] * np.clip(snr_short / snr_div, *scale['tp_short_clip']),
            'sl_short': params_base['sl_short'] * vol_quality * np.clip(scale['sl_short_numerator'] / snr_short, *scale['sl_short_clip']),
            'max_hold': max(scale['min_hold_hours'], params_base['max_hold_hours'] * vol_quality),
            'trailing_start_long': params_base['trailing_stop_start_ratio'] * vol_quality,
            'trailing_start_short': params_base['trailing_stop_start_ratio'] * scale['short_trailing_multiplier'] * vol_quality,
        }

    def run_backtest_simulation(self, price_data: pd.DataFrame,
                                dynamic_params: dict,
                                use_sticky: bool = None,
                                use_adx: bool = None,
                                only_selected_zone: bool = None):
        """
        Runs the trade execution loop with optional ADX and Sticky filters.
        """
        use_sticky = (
            use_sticky
            if use_sticky is not None
            else self.filter_settings['use_sticky']
        )
        use_adx = (
            use_adx
            if use_adx is not None
            else self.filter_settings['use_adx']
        )
        only_selected_zone = (
            only_selected_zone
            if only_selected_zone is not None
            else self.filter_settings['only_selected_zone']
        )

        trades_list = []
        is_in_position = False
        active_position = {}
        round_trip_cost = (
            self.execution_costs['commission_rate'] +
            self.execution_costs['slippage_rate']
            ) * 2

        adx_boost_mult = self.scaling_parameters.get('adx_boost_threshold_multiplier', 1.2)
        hold_boost_ratio = self.scaling_parameters.get('max_hold_boost_ratio', 1.5)
        entry_threshold = self.risk_parameters.get('entry_probability_threshold', 0.5)
        adx_thr = self.trading_parameters['adx_threshold']

        for i in range(len(price_data) - 1):
            curr_row = price_data.iloc[i]
            next_row = price_data.iloc[i+1]
            next_timestamp = price_data.index[i+1]

            if is_in_position:
                exit_result = self._handle_exit_logic(active_position,
                                                      next_row,
                                                      next_timestamp,
                                                      round_trip_cost)
                if exit_result:
                    trades_list.append(exit_result)
                    is_in_position = False
            else:
                # 1. Zone Filter
                zone_passed = curr_row['in_zone'] if only_selected_zone else True

                # 2. Signal Filter (Sticky vs Raw)
                signal_passed = (
                    (curr_row['sticky_signal'] == 1)
                    if use_sticky
                    else (curr_row['regime_prob'] > entry_threshold)
                )

                # 3. ADX Filter
                adx_val = curr_row['ADX']
                adx_passed = (adx_val > adx_thr) if use_adx else True

                # Final Entry Decision
                if zone_passed and signal_passed and adx_passed:
                    entry_p = next_row['Open']

                    # Apply ADX-based Holding Boost (Always calculated based on ADX value if trigger is met)
                    current_max_hold = dynamic_params['max_hold']
                    if adx_val > adx_thr * adx_boost_mult:
                        current_max_hold *= hold_boost_ratio

                    # Position Initialization
                    if curr_row['Close'] > curr_row['dynamic_resistance']:
                        sl_price = min(entry_p * (1 - dynamic_params['sl_long']), curr_row['dynamic_resistance'])
                        active_position = {
                            'position_type': 'Long', 'entry_price': entry_p, 'entry_time': next_timestamp,
                            'tp_price': entry_p * (1 + dynamic_params['tp_long']), 'sl_price': sl_price,
                            'hwm': entry_p, 'tp_target': dynamic_params['tp_long'], 'sl_target': dynamic_params['sl_long'],
                            'trail_start': dynamic_params['trailing_start_long'], 'max_hold': current_max_hold
                        }
                        is_in_position = True
                    elif curr_row['Close'] < curr_row['dynamic_support']:
                        sl_price = max(entry_p * (1 + dynamic_params['sl_short']), curr_row['dynamic_support'])
                        active_position = {
                            'position_type': 'Short', 'entry_price': entry_p, 'entry_time': next_timestamp,
                            'tp_price': entry_p * (1 - dynamic_params['tp_short']), 'sl_price': sl_price,
                            'lwm': entry_p, 'tp_target': dynamic_params['tp_short'], 'sl_target': dynamic_params['sl_short'],
                            'trail_start': dynamic_params['trailing_start_short'], 'max_hold': current_max_hold
                        }
                        is_in_position = True

        results_df = pd.DataFrame(trades_list)
        if not results_df.empty:
            results_df = results_df.sort_values('entry_time').reset_index(drop=True)
            results_df['equity'] = (1 + results_df['PnL']).cumprod()
            results_df['drawdown'] = (
                (results_df['equity'] - results_df['equity'].cummax()) / results_df['equity'].cummax()
            )

        return results_df

    def _handle_exit_logic(
        self,
        pos: dict,
        next_row: pd.Series,
        next_time: pd.Timestamp,
        cost: float,
        ):
        """
        Handles granular exit triggers: TP, SL, Break-even, Trailing stops, and Time-out.
        """
        position_type = pos['position_type']
        entry_price = pos['entry_price']
        entry_time = pos['entry_time']

        if position_type == 'Long':
            pos['hwm'] = max(pos['hwm'], next_row['High'])

            break_even_ratio = self.risk_parameters['break_even_trigger_ratio_long']
            if next_row['High'] >= entry_price * (1 + pos['tp_target'] * break_even_ratio):
                pos['sl_price'] = max(pos['sl_price'], entry_price * 1.0005)

            if pos['hwm'] >= entry_price * (1 + pos['trail_start']):
                pos['sl_price'] = max(pos['sl_price'], pos['hwm'] * (1 - pos['sl_target']))

            if next_row['Low'] <= pos['sl_price']:
                return {'PnL': (pos['sl_price'] - entry_price) / entry_price - cost,
                        'entry_time': entry_time, 'exit_time': next_time, 'type': 'Long',
                        'result': 'StopLoss'}
            elif next_row['High'] >= pos['tp_price']:
                return {'PnL': pos['tp_target'] - cost, 'entry_time': entry_time,
                        'exit_time': next_time, 'type': 'Long', 'result': 'Win'}

        elif position_type == 'Short':
            pos['lwm'] = min(pos['lwm'], next_row['Low'])

            break_even_ratio = self.risk_parameters['break_even_trigger_ratio_short']
            if next_row['Low'] <= entry_price * (1 - pos['tp_target'] * break_even_ratio):
                pos['sl_price'] = min(pos['sl_price'], entry_price * 0.9995)

            if pos['lwm'] <= entry_price * (1 - pos['trail_start']):
                pos['sl_price'] = min(pos['sl_price'], pos['lwm'] * (1 + pos['sl_target']))

            if next_row['High'] >= pos['sl_price']:
                return {'PnL': (entry_price - pos['sl_price']) / entry_price - cost,
                        'entry_time': entry_time, 'exit_time': next_time, 'type': 'Short',
                        'result': 'StopLoss'}
            elif next_row['Low'] <= pos['tp_price']:
                return {'PnL': pos['tp_target'] - cost, 'entry_time': entry_time,
                        'exit_time': next_time, 'type': 'Short', 'result': 'Win'}

        if (next_time - entry_time).total_seconds() / 3600 > pos['max_hold']:
            pnl = (
                (next_row['Close'] - entry_price) / entry_price
                if position_type == 'Long'
                else (entry_price - next_row['Close']) / entry_price
            )
            return {'PnL': pnl - cost, 'entry_time': entry_time, 'exit_time': next_time,
                    'type': position_type, 'result': 'TimeOut'}

        return None
