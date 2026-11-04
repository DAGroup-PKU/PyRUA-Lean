export const LEADERBOARD = {
  "checked": "2026-11-03",
  "boardUpdated": "2026-10-30",
  "repository": "https://github.com/DAGroup-PKU/PyRUA-Lean/blob/main/integrations/xpolicylab/PyRUA_RSI/eval/robodojo_summary.md",
  "leaderboard": "https://robodojo-benchmark.com/leaderboard?bench=sim",
  "selection": "Top three across all entries by Average SR (also the top three by Score); highest pure WAM and VLA entries; GPT-6-Astra. Deduplicated.",
  "models": [
    {
      "name": "PyRUA-REAL (ours)",
      "kind": "ours",
      "tag": "Fixed programs \u00b7 seed 0",
      "values": {
        "average": {
          "sr": 84.42,
          "score": 88.53
        },
        "generalization": {
          "sr": 72.33,
          "score": 78.08
        },
        "precision": {
          "sr": 91.5,
          "score": 92.55
        },
        "longHorizon": {
          "sr": 80,
          "score": 88.22
        },
        "memory": {
          "sr": 95,
          "score": 95.55
        },
        "open": {
          "sr": 83.25,
          "score": 88.24
        }
      }
    },
    {
      "name": "Q0-Preview",
      "values": {
        "average": {
          "score": 74.2,
          "sr": 68.87
        },
        "generalization": {
          "score": 64.0,
          "sr": 58.67
        },
        "precision": {
          "score": 65.3,
          "sr": 58.5
        },
        "longHorizon": {
          "score": 69.8,
          "sr": 58.5
        },
        "memory": {
          "score": 93.68,
          "sr": 93.67
        },
        "open": {
          "score": 78.2,
          "sr": 75.0
        }
      },
      "kind": "baseline",
      "tag": "Overall top 1"
    },
    {
      "name": "GO-3 Nano Preview",
      "values": {
        "average": {
          "score": 61.9,
          "sr": 56.42
        },
        "generalization": {
          "score": 36.45,
          "sr": 33.665
        },
        "precision": {
          "score": 78.65,
          "sr": 76.0
        },
        "longHorizon": {
          "score": 68.21,
          "sr": 53.5
        },
        "memory": {
          "score": 64.5,
          "sr": 63.67
        },
        "open": {
          "score": 61.71,
          "sr": 55.25
        }
      },
      "kind": "baseline",
      "tag": "Overall top 2"
    },
    {
      "name": "PyRUA-Lean",
      "values": {
        "average": {
          "score": 55.26,
          "sr": 50.35
        },
        "generalization": {
          "score": 48.269999999999996,
          "sr": 43.16666666666667
        },
        "precision": {
          "score": 44.77,
          "sr": 38.25
        },
        "longHorizon": {
          "score": 54.3,
          "sr": 43.75
        },
        "memory": {
          "score": 68.4,
          "sr": 68.33
        },
        "open": {
          "score": 60.56,
          "sr": 58.25
        }
      },
      "kind": "baseline",
      "tag": "Overall top 3"
    },
    {
      "name": "VPP2",
      "values": {
        "average": {
          "score": 39.26,
          "sr": 32.26
        },
        "generalization": {
          "score": 36.385,
          "sr": 29.5
        },
        "precision": {
          "score": 43.03,
          "sr": 35.17
        },
        "longHorizon": {
          "score": 56.28,
          "sr": 37.0
        },
        "memory": {
          "score": 56.01,
          "sr": 55.56
        },
        "open": {
          "score": 4.61,
          "sr": 4.08
        }
      },
      "kind": "baseline",
      "tag": "Top WAM"
    },
    {
      "name": "WALL-OSS-1.0-lite",
      "values": {
        "average": {
          "score": 42.59,
          "sr": 34.97
        },
        "generalization": {
          "score": 46.66,
          "sr": 37.94
        },
        "precision": {
          "score": 38.16,
          "sr": 29.08
        },
        "longHorizon": {
          "score": 49.94,
          "sr": 32.25
        },
        "memory": {
          "score": 57.42,
          "sr": 56.0
        },
        "open": {
          "score": 20.76,
          "sr": 19.58
        }
      },
      "kind": "baseline",
      "tag": "Top VLA"
    },
    {
      "name": "GPT-6-Astra",
      "values": {
        "average": {
          "score": 28.97,
          "sr": 22.48
        },
        "generalization": {
          "score": 33.36,
          "sr": 30.5
        },
        "precision": {
          "score": 12.65,
          "sr": 4.0
        },
        "longHorizon": {
          "score": 21.45,
          "sr": 8.25
        },
        "memory": {
          "score": 43.04,
          "sr": 38.67
        },
        "open": {
          "score": 34.36,
          "sr": 31.0
        }
      },
      "kind": "astra",
      "tag": "Direct control"
    }
  ],
  "localRevision": "User-supplied Precision and Average SR/Score, 2026-11-03"
};
