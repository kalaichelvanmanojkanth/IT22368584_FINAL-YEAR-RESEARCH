# RACE-Dengue analysis

Study: niteroi_zone_v3

Area: zone_1

Evidence status: INSUFFICIENT_EVIDENCE

This is a research decision-support prototype. It is not a clinical diagnostic system. Intervention-effect estimates are conditional on the data and causal assumptions and should receive domain review.

- Unvalidated gap between training history and intervention window
- Too few independent calendar calibration blocks
- Single donor; donor-removal stability cannot be established
- Pre-fitted CITS variant; not a replication of the publication's full segmented regression
- Base estimator has no native calibrated interval; ensemble calibration must be assessed separately
- Treatment effect coefficient is not fitted; effect is an observed-minus-Y(0) contrast
- Negative-binomial alpha is fixed by config, not estimated or tuned on the post period
- Only one donor: no learned donor diversity or leave-one-donor-out estimate
- Severe donor concentration: largest weight exceeds configured diagnostic threshold
- Base estimator has no native calibrated interval
- International Wolbachia estimates do not establish effects in Sri Lanka or effects of fogging/source reduction

## Full pipeline result and provenance

```json
{
  "study_id": "niteroi_zone_v3",
  "location_id": "zone_1",
  "intervention_type": "wMel Wolbachia mosquito deployments",
  "effect_absolute": null,
  "effect_percent": null,
  "interval_lower": null,
  "interval_upper": null,
  "reliability_score": 0.54974718431531,
  "decision": "INSUFFICIENT_EVIDENCE",
  "model_weights": {
    "cits": 0.49548392172957734,
    "synthetic_control": 0.5045160782704227
  },
  "diagnostics": {
    "pre_rmse": 1.1253343638139295,
    "model_disagreement": 0.029713846304464022,
    "missing_rate": 0.0,
    "extrapolation_features": 0.46008242056868826,
    "gap_months": 28.0,
    "horizon": 6.0,
    "effective_donor_count": 1.0,
    "predicted_error": 0.17166823156433253,
    "predicted_errors_by_model": {
      "cits": 0.19901018427586217,
      "synthetic_control": 0.14481577127466888
    },
    "models": {
      "cits": {
        "model_name": "cits",
        "treated_unit": "zone_1",
        "control_units": [
          "control"
        ],
        "training_start": "2007-01-01",
        "training_end": "2014-12-01",
        "intervention_date": "2017-05-01",
        "config": {
          "min_pre_months": 36,
          "min_post_months": 6,
          "horizon_months": 6,
          "seasonal_harmonics": 2,
          "cits_family": "negative_binomial",
          "nb_alpha": 1.0,
          "scm_outcome": "dengue_incidence",
          "donor_concentration_warning": 0.8,
          "optimizer_max_iterations": 2000,
          "optimizer_tolerance": 1e-10,
          "holdout_months": 12
        },
        "diagnostics": {
          "pre_fit_metrics": {
            "mae": 11.160368759019079,
            "rmse": 24.466339954542068,
            "mase": 0.5833671286606115
          },
          "residual_variance": 596.1642008184075,
          "residual_autocorrelation": 0.6281538846564905,
          "available_history": 96,
          "missing_rate": 0.0,
          "residuals": [
            -10.056712097992035,
            -10.203631221458686,
            12.135075549660428,
            18.78253890327204,
            31.00531287357039,
            5.341040848187152,
            0.9757073480966438,
            -0.8933708814970058,
            1.46006251397962,
            0.6977617992188188,
            -1.8332052766755123,
            1.0815228407120934,
            -19.43458224815226,
            -25.795904175107268,
            -76.59715041084442,
            -120.88368526364962,
            -16.273818083799746,
            -4.145786847232834,
            -4.018680008439151,
            -1.343988101270412,
            -0.7370910396643735,
            4.172863870478291,
            0.472816680904395,
            -3.0590217338793613,
            3.2864906661053777,
            0.7398450426874454,
            14.323137152434109,
            0.757764019732833,
            3.067340882596275,
            1.7399985300794931,
            -1.0146032896947148,
            -0.7142161614248812,
            -0.38223700404147887,
            -0.24332296326992883,
            -0.31671806742430486,
            4.040876056442686,
            -0.40679157217001805,
            -3.6823987277031485,
            -7.641219376886818,
            -5.492950622093357,
            -10.656436495417715,
            -0.039303870159605125,
            -2.031566727058425,
            2.112254231807125,
            0.5328804501338089,
            0.4019782945202075,
            -1.9263225443730252,
            -8.669188138430309,
            -4.672346279966019,
            -0.38059579624150075,
            35.1026592324205,
            36.04548483932187,
            27.751183452314052,
            6.945271776609793,
            0.7837603669824658,
            -0.6247634376173368,
            -1.6320548171915803,
            3.705049962122443,
            -2.2668306164950964,
            6.287177882048448,
            63.77899877662851,
            41.22219928185791,
            55.371529407953524,
            -27.89233373544448,
            -4.810248504594512,
            -3.369306978616205,
            -0.9156975989719882,
            -1.0945810151900224,
            1.994305022345229,
            0.2398550713349763,
            0.5768423019856215,
            0.8751366468165207,
            -18.258539964343964,
            -46.46435262407931,
            -80.73782724656292,
            -103.7904119905042,
            -20.78880418550358,
            1.9142791728097794,
            -3.469786182174101,
            1.141299420531161,
            -0.8737582637145105,
            -1.0317220229647333,
            -1.2831882448392335,
            -3.3219333818891696,
            -3.352372253412394,
            -0.6055344312936777,
            -1.3907693509475565,
            -1.2814638485311265,
            -0.6944477948418424,
            -0.4989138191370877,
            1.3516800974933632,
            0.33106066841852744,
            1.9993550306522558,
            -1.6787998598268312,
            0.0003790525832401137,
            -1.1753376532767676
          ],
          "coefficients": {
            "intercept": -10.928972367812284,
            "trend_years": -0.10353033650792218,
            "sin_1": 0.20736776821143013,
            "cos_1": 0.26852386208722934,
            "sin_2": -0.07680520594981993,
            "cos_2": -0.0036263028478260052,
            "log1p_rate_control": 0.9656867592761218
          },
          "converged": true,
          "design_condition_number": 21.67304716786109,
          "family": "negative_binomial",
          "nb_alpha": 1.0,
          "counterfactual_intervention_indicator": 0
        },
        "assumptions": [
          "Pre-period treated/control relationship remains stable without intervention",
          "Control outcomes are unaffected by intervention and spillover",
          "Population denominators and surveillance definitions are comparable over time",
          "Post control outcomes are permitted for retrospective evaluation, not prospective forecasting"
        ],
        "warnings": [
          "Pre-fitted CITS variant; not a replication of the publication's full segmented regression",
          "Base estimator has no native calibrated interval; ensemble calibration must be assessed separately",
          "Treatment effect coefficient is not fitted; effect is an observed-minus-Y(0) contrast",
          "Negative-binomial alpha is fixed by config, not estimated or tuned on the post period"
        ]
      },
      "synthetic_control": {
        "model_name": "synthetic_control",
        "treated_unit": "zone_1",
        "control_units": [
          "control"
        ],
        "training_start": "2007-01-01",
        "training_end": "2014-12-01",
        "intervention_date": "2017-05-01",
        "config": {
          "min_pre_months": 36,
          "min_post_months": 6,
          "horizon_months": 6,
          "seasonal_harmonics": 2,
          "cits_family": "negative_binomial",
          "nb_alpha": 1.0,
          "scm_outcome": "dengue_incidence",
          "donor_concentration_warning": 0.8,
          "optimizer_max_iterations": 2000,
          "optimizer_tolerance": 1e-10,
          "holdout_months": 12
        },
        "diagnostics": {
          "pre_fit_metrics": {
            "mae": 11.246130317113222,
            "rmse": 24.743593996404556,
            "mase": 0.5878499979075984
          },
          "residual_variance": 616.3878554679402,
          "residual_autocorrelation": 0.6579991908877357,
          "available_history": 96,
          "missing_rate": 0.0,
          "residuals": [
            -3.548916647251743,
            1.298517894585924,
            72.2421907202638,
            71.82759547662145,
            51.08460725102601,
            10.10034230004122,
            2.045610136382373,
            -0.5854047563576408,
            1.8291904872847184,
            1.2072976218211795,
            1.658380974569437,
            8.08209824548827,
            0.42622627645656763,
            -7.186544561730496,
            -3.1582644850266206,
            -36.811627448520596,
            1.544812631050732,
            -1.1069553217799601,
            -3.567160701804691,
            -1.1890535672682305,
            -0.5854047563576408,
            4.838312514561193,
            2.441961325471783,
            -0.16168748543880707,
            8.667503001845912,
            8.911288732772988,
            19.06385419093532,
            3.640136920016488,
            4.216419649097654,
            2.2072976218211795,
            -0.7927023781788204,
            -0.5945267836341153,
            -0.1981755945447051,
            0.0,
            0.2072976218211796,
            4.60364881091059,
            0.0,
            -1.7612543716066944,
            -4.885296195919535,
            -1.7085714796826519,
            -6.5675262296401655,
            0.5658536149050555,
            -2.460487831056967,
            1.3891288986681691,
            0.026341445962021925,
            0.513170722981011,
            -1.3364460067441257,
            -6.071358932388799,
            -1.7969338378010935,
            4.255749054122951,
            44.378118263700685,
            49.32209014230642,
            31.271079865117485,
            6.778257797336338,
            0.7255749054122951,
            -1.1240418243128416,
            -3.0977003783508197,
            3.1767247162368855,
            -1.973658554037978,
            7.627874527061476,
            67.43247377035983,
            45.246411033890574,
            61.0943551488806,
            -23.017701093907917,
            -8.988014418475686,
            -8.736585540379778,
            -4.469825851289343,
            -2.885296195919535,
            1.026341445962022,
            -0.12404182431284161,
            0.7255749054122951,
            1.3891288986681691,
            -16.187735351204374,
            -44.70592391839071,
            -88.86467026234112,
            -125.88501891754098,
            -28.01268324970259,
            -0.40780493913292304,
            -5.310104560782103,
            0.7255749054122951,
            -1.274425094587705,
            -1.274425094587705,
            -1.274425094587705,
            -3.186062736469262,
            -3.1333798445452175,
            -0.31010456078210336,
            -1.1240418243128416,
            -1.0620209121564208,
            -0.6992334594502734,
            -0.42480836486256834,
            1.1503832702748633,
            -0.062020912156420804,
            1.3007665405497266,
            -2.54885018917541,
            -0.062020912156420804,
            -1.0620209121564208
          ],
          "donor_weights": {
            "control": 1.0
          },
          "max_donor_weight": 1.0,
          "donor_count": 1,
          "effective_donor_count": 1.0,
          "optimisation_converged": true,
          "optimisation_message": "Single-donor simplex: analytic unique solution w=1",
          "matching_outcome": "dengue_incidence",
          "matching_rmse": 106.24016572106345,
          "leave_one_donor_sensitivity": null
        },
        "assumptions": [
          "The untreated trajectory lies sufficiently close to the donor convex hull",
          "Donors are unaffected by intervention/spillover and share surveillance changes",
          "Pre-period donor weights remain applicable after intervention",
          "Incidence matching is converted to case counts using supplied population"
        ],
        "warnings": [
          "Only one donor: no learned donor diversity or leave-one-donor-out estimate",
          "Severe donor concentration: largest weight exceeds configured diagnostic threshold",
          "Base estimator has no native calibrated interval"
        ]
      }
    }
  },
  "assumptions": [
    "Pre-period treated/control relationship remains stable without intervention",
    "Control outcomes are unaffected by intervention and spillover",
    "Population denominators and surveillance definitions are comparable over time",
    "Post control outcomes are permitted for retrospective evaluation, not prospective forecasting",
    "The untreated trajectory lies sufficiently close to the donor convex hull",
    "Donors are unaffected by intervention/spillover and share surveillance changes",
    "Pre-period donor weights remain applicable after intervention",
    "Incidence matching is converted to case counts using supplied population"
  ],
  "warnings": [
    "Unvalidated gap between training history and intervention window",
    "Too few independent calendar calibration blocks",
    "Single donor; donor-removal stability cannot be established",
    "Pre-fitted CITS variant; not a replication of the publication's full segmented regression",
    "Base estimator has no native calibrated interval; ensemble calibration must be assessed separately",
    "Treatment effect coefficient is not fitted; effect is an observed-minus-Y(0) contrast",
    "Negative-binomial alpha is fixed by config, not estimated or tuned on the post period",
    "Only one donor: no learned donor diversity or leave-one-donor-out estimate",
    "Severe donor concentration: largest weight exceeds configured diagnostic threshold",
    "Base estimator has no native calibrated interval",
    "International Wolbachia estimates do not establish effects in Sri Lanka or effects of fogging/source reduction"
  ],
  "trajectories": [
    {
      "date": "2017-05-01",
      "observed": 1.0,
      "counterfactual": 5.737166762492361,
      "lower": 1.7075053658077595,
      "upper": 9.766828159176963,
      "cits": 4.441805145907678,
      "synthetic_control": 7.009338020232377
    },
    {
      "date": "2017-06-01",
      "observed": 0.0,
      "counterfactual": 4.090128767531554,
      "lower": 0.060467370846952306,
      "upper": 8.119790164216155,
      "cits": 2.8479140844655344,
      "synthetic_control": 5.310104560782103
    },
    {
      "date": "2017-07-01",
      "observed": 0.0,
      "counterfactual": 0.9667649313918505,
      "lower": 0.0,
      "upper": 4.996426328076452,
      "cits": 0.6534964434179346,
      "synthetic_control": 1.274425094587705
    },
    {
      "date": "2017-08-01",
      "observed": 0.0,
      "counterfactual": 0.9241714411071684,
      "lower": 0.0,
      "upper": 4.95383283779177,
      "cits": 0.5675330278221233,
      "synthetic_control": 1.274425094587705
    },
    {
      "date": "2017-09-01",
      "observed": 1.0,
      "counterfactual": 1.2207213098519702,
      "lower": 0.0,
      "upper": 5.250382706536572,
      "cits": 0.7334863814664532,
      "synthetic_control": 1.6992334594502734
    },
    {
      "date": "2017-10-01",
      "observed": 1.0,
      "counterfactual": 0.8277168331997486,
      "lower": 0.0,
      "upper": 4.85737822988435,
      "cits": 0.5891416345831123,
      "synthetic_control": 1.0620209121564208
    }
  ],
  "exploratory_contrast": {
    "effect_absolute": -10.766670045574651,
    "estimated_cases_averted": 10.766670045574651,
    "effect_percent": -78.2082377941181,
    "interval_lower": -34.94463842568226,
    "interval_upper": 1.2320272633452882,
    "label": "Unvalidated exploratory model contrast; do not use as a supported intervention conclusion"
  },
  "explanation": {
    "label": "Model Reliability Score",
    "definition": "1 / (1 + predicted normalized MAE / median training normalized MAE)",
    "interpretation": "Relative predicted model error; not a probability of causality or clinical risk",
    "factors": {
      "predicted_error": 0.17166823156433253,
      "pre_rmse": 1.1253343638139295,
      "model_disagreement": 0.029713846304464022,
      "missing_rate": 0.0,
      "extrapolation_features": 0.46008242056868826,
      "gap_months": 28.0,
      "effective_donor_count": 1.0
    },
    "decision": "INSUFFICIENT_EVIDENCE",
    "reasons": [
      "Unvalidated gap between training history and intervention window",
      "Too few independent calendar calibration blocks",
      "Single donor; donor-removal stability cannot be established"
    ],
    "calibration_calendar_years": 1,
    "unavailable_signals": [
      "Single-donor removal stability",
      "Validated real-world causal ground truth"
    ],
    "interval_warning": "Empirical calendar-block intervals; exact coverage is not established"
  },
  "provenance": {
    "reliability_run": "20261021T215308Z_52d8d5dc",
    "dataset_sha256": "5f4d4e8dee470de2a1f9bbc99290127a11ebc6d24e5e05f4eaca743e31ecb9c7",
    "train_end": "2014-12-01",
    "monitoring_start": "2017-05-01",
    "zone_wide_release_start": "2017-04-01"
  },
  "dashboard_context": {
    "run_id": "20261021T215308Z_52d8d5dc",
    "dataset_sha256": "5f4d4e8dee470de2a1f9bbc99290127a11ebc6d24e5e05f4eaca743e31ecb9c7",
    "limitations": "This is a research decision-support prototype. It is not a clinical diagnostic system. Intervention-effect estimates are conditional on the data and causal assumptions and should receive domain review."
  }
}
```
