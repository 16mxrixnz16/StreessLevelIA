# StreessLevelIA
# Menta

## How to use

```bash
pip install -r requirements.txt
```

## How to run

```bash
python main.py
```

## Calculate Stress

Give some weights to each emotion, and calculate the average.

```

Confusion,    Confusion,    Confusion,  Calm, Calm, Calm, Calm,               Surprised
  0.8            0.8           0.8      0.1      0.1      0.1      0.1      0.7


result_stress_average = (0.8 + 0.8 + 0.8 + 0.1 + 0.1 + 0.1 + 0.1 + 0.7) / 8


result_stress = 1 - result_stress_average

```


## Refs

- https://www.gradio.app/docs/gradio/audio