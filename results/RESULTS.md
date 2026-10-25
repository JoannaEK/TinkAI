### promt_0 / promt_1: финальные модели

| версия | horizon | поиск | признаков | ROC | PR |
|---|---|---|---|---|---|
| v1_grid5_h7 | 7 | grid5 | 7 | 0.907672 | 0.713992 |
| v2_grid5_h14 | 14 | grid5 | 4 | 0.883246 | 0.680747 |
| v3_grid10_h14 | 14 | grid10 | 8 | 0.891614 | 0.695794 |
| v4_grid10_exp4_h14 | 14 | grid10_exp4 | 8 | 0.893138 | 0.696803 |
| v5_stoch_fixed10_h14 | 14 | stoch_fixed10 | 12 | 0.893456 | 0.705081 |
| v6_stoch_adaptive_h14 | 14 | stoch_adaptive | 13 | 0.899189 | 0.714743 |
| v7_mean_exp3_h14 | 14 | mean_then_exp3 | 8 | 0.897364 | 0.711631 |
| v8_perfeature_h14 | 14 | perfeature_adaptive | 19 | 0.897482 | 0.713319 |

Признаки финальных моделей:

| версия | признаки (оптимальное окно, дней) |
|---|---|
| v1_grid5_h7 | user_tx_cnt (n=719); user_avg_active_day_gap (n=719); user_avg_tx (n=719); vcat_tx_cnt (x=7); vcat_avg_active_day_gap (x=363); vcat_repeat_purchase_rate (x=7); vcat_avg_tx (x=363) |
| v2_grid5_h14 | user_tx_cnt (n=719); user_avg_active_day_gap (n=719); user_avg_tx (n=719); vi_tx_cnt (q=185) |
| v3_grid10_h14 | user_tx_cnt (n=640); user_avg_active_day_gap (n=719); user_avg_tx (n=719); vcat_tx_cnt (x=86); vcat_avg_active_day_gap (x=7); vcat_repeat_purchase_rate (x=561); vcat_avg_tx (x=403); vi_tx_cnt (q=165) |
| v4_grid10_exp4_h14 | user_tx_cnt (n=620); user_avg_active_day_gap (n=719); user_avg_tx (n=719); vcat_tx_cnt (x=86); vcat_avg_active_day_gap (x=7); vcat_repeat_purchase_rate (x=571); vcat_avg_tx (x=393); vi_tx_cnt (q=165) |
| v5_stoch_fixed10_h14 | user_tx_cnt (n=353); user_avg_active_day_gap (n=363); user_avg_tx (n=363); item_tx_cnt (m=373); item_avg_active_day_gap (m=363); item_repeat_purchase_rate (m=353); item_avg_tx (m=373); vcat_tx_cnt (x=363); vcat_avg_active_day_gap (x=363); vcat_repeat_purchase_rate (x=353); vcat_avg_tx (x=363); vi_tx_cnt (q=373) |
| v6_stoch_adaptive_h14 | user_tx_cnt (n=443); user_avg_active_day_gap (n=363); user_avg_tx (n=363); item_tx_cnt (m=363); item_avg_active_day_gap (m=363); item_repeat_purchase_rate (m=363); item_avg_tx (m=453); vcat_tx_cnt (x=363); vcat_avg_active_day_gap (x=363); vcat_repeat_purchase_rate (x=283); vcat_avg_tx (x=363); vi_tx_cnt (q=363); vc_tx_cnt (s=363) |
| v7_mean_exp3_h14 | user_tx_cnt (n=363); user_avg_active_day_gap (n=283); user_avg_tx (n=443); item_tx_cnt (m=363); item_avg_active_day_gap (m=363); item_repeat_purchase_rate (m=363); item_avg_tx (m=363); vi_tx_cnt (q=363) |
| v8_perfeature_h14 | user_tx_cnt (n=483); user_avg_active_day_gap (n=363); user_avg_tx (n=363); item_tx_cnt (m=363); item_avg_active_day_gap (m=363); item_repeat_purchase_rate (m=363); item_avg_tx (m=363); cat_tx_cnt (k=363); cat_avg_active_day_gap (k=363); cat_repeat_purchase_rate (k=323); cat_avg_tx (k=363); vcat_tx_cnt (x=363); vcat_avg_active_day_gap (x=443); vcat_repeat_purchase_rate (x=363); vcat_avg_tx (x=363); vi_tx_cnt (q=363); vc_tx_cnt (s=263); ic_tx_cnt (z=363); ii_co_purchase (u=283) |

### Опорная модель (все 19 признаков, окно 363)

| horizon | признаков | пар | positives | ROC | PR |
|---|---|---|---|---|---|
| 7.0 | 19.0 | 34165.0 | 6833.0 | 0.913691 | 0.775219 |
| 14.0 | 19.0 | 62635.0 | 12527.0 | 0.886328 | 0.693997 |

### promt_2: long_grid_search

ROC = **0.8927**, PR = **0.7229**, итераций: 8, horizon = 7.

| признак | групповой параметр | оптимальное окно, дней |
|---|---|---|
| user_avg_active_day_gap | n | 719 |
| vi_tx_cnt | q | 252 |
| ic_tx_cnt | z | 7 |

### promt_3: short_stochastic_search на подвыборках

| способ | метод | выборка | пользователей | добавлено | пар | признаков | ROC (подвыборка) | PR (подвыборка) | ROC (полный) | PR (полный) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | uniform_1pct | 0 | 59 | 0 | 60 | 5 | 1.0 | 1.0 | 0.912981 | 0.750555 |
| 1 | uniform_1pct | 1 | 59 | 0 | 440 | 14 | 0.977778 | 0.922122 | 0.921784 | 0.785611 |
| 1 | uniform_1pct | 2 | 59 | 0 | 530 | 14 | 0.992997 | 0.976117 | 0.913123 | 0.771779 |
| 2 | typical_top10_uniform | 0 | 42 | 0 | 70 | 7 | 0.818182 | 0.452381 | 0.902458 | 0.719565 |
| 2 | typical_top10_uniform | 1 | 71 | 29 | 75 | 9 | 0.972222 | 0.916667 | 0.809396 | 0.566561 |
| 2 | typical_top10_uniform | 2 | 42 | 0 | 85 | 2 | 1.0 | 1.0 | 0.812342 | 0.598946 |
| 3 | typical_weighted | 0 | 62 | 15 | 85 | 8 | 1.0 | 1.0 | 0.91536 | 0.730682 |
| 3 | typical_weighted | 1 | 62 | 21 | 85 | 11 | 0.833333 | 0.666667 | 0.844324 | 0.636636 |
| 3 | typical_weighted | 2 | 62 | 19 | 85 | 9 | 1.0 | 1.0 | 0.922386 | 0.776505 |
