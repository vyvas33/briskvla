# BriskVLA

CMU 10-623 course project built on LeRobot + SmolVLA, reducing inference latency on LIBERO through shallower action experts, FFN fusion, and fewer denoising steps.
Original LeRobot README: [README_LEROBOT.md](README_LEROBOT.md)

## Setup (Linux + NVIDIA GPU; LIBERO doesn't install on macOS)

```bash
git clone https://github.com/vyvas33/briskvla.git
cd briskvla
git switch briskvla
uv sync --locked --python 3.12 --extra briskvla --extra libero   # omit --extra libero on macOS or distill-only VMs
uv run wandb login
uv run hf auth login
export MUJOCO_GL=egl   # headless servers
```

Use Python 3.12 (required for `draccus` config parsing) and install system `ffmpeg` (`sudo apt install ffmpeg`) for fast video decoding.

## Branch Workflow (Common Ground)

- `main`: untouched copy of `huggingface/lerobot`. Do not commit here.
- `briskvla`: **our single shared common ground**. All standardized code, W&B instrumentation, and verified methods live here.
- Personal experiments: always branch from `briskvla` (`git switch -c <name>/<topic> origin/briskvla`), then merge back via PR to `briskvla`.

## Team Standards

1. **Mandatory W&B logging**: Every training, distillation, LIBERO eval, and latency run logs automatically to Weights & Biases (`entity: djonnala-carnegie-mellon-university`, `project: briskVLA`). Runs fail fast if unauthenticated (set `WANDB_API_KEY` or run `uv run wandb login`; use `WANDB_MODE=offline` on air-gapped machines).
2. **Standard LIBERO eval settings** (enforced by [project/briskvla/run_eval.sh](project/briskvla/run_eval.sh)): `n_action_steps=10`, `seed=1000`, `episodes=10` per task (100 per suite), `batch_size=5`, `empty_cameras=1`, evaluated at `K=10` and `K=3` denoising steps.
3. **Comparable hardware**: Measure inference latency only on the fixed **RTX 3080 Laptop GPU**, and LIBERO task success on Google Colab (or equivalent CUDA GPU with fixed `run_eval.sh` settings).

## Model Naming

Every model and run name makes its training method, expert depth, layer placement, and compression immediately clear (`_K<steps>` is appended to eval folders for denoising steps):

`teacher` or `<role>_<layers>L_<placement>[_<modification>][_<variant>]`

| Component | Meaning | Values |
|---|---|---|
| `<role>` | Whether teacher distillation was used | `student` = distilled from teacher · `shallow` = **not distilled** (GT flow matching only, or untrained) |
| `<layers>L` | Number of action-expert transformer layers | e.g. `8L`, `4L` (teacher has `16L`; distinct from denoising steps `K10`, `K3`) |
| `<placement>` | Which of the 16 VLM depths host an expert layer | `spread` (`0,1,4,5,8,9,12,13`) · `early` (`0–7`) · `late` (`8–15`) · `alternating` (`0,3,4,7,8,11,12,15`) |
| `<modification>` | Optional structural change (omitted for plain drop) | `ffn_fused` (merged pair FFNs stacked in parallel) |
| `<variant>` | Optional training/eval qualifier | `untrained` · `shallow_pi` (teacher velocity/hidden + GT, no output loss) · `steps3` |

Legacy CLI layout aliases (`pairs`, `first8`, `last8`, `alt`, `alternate`) are still accepted by `--layout`. Every model, its architecture, and its results are listed in [Models & Results](#models--results).

## Quick Eval

```bash
# Latency only (RTX 3080 default):
project/briskvla/run_eval.sh teacher teacher

# Full LIBERO success eval (Colab):
SUCCESS=1 LATENCY=0 project/briskvla/run_eval.sh student_8L_spread path/to/checkpoint
```

## Models & Results

LIBERO success rate (%), 100 episodes per suite, seed 1000, `n_action_steps=10`. Each model has two rows: `K=10` and `K=3` denoising steps at eval. Each trained 8-layer (`8L`) model ran for 5000 steps at batch size 8. `K=1` results and per-variant details: [project/briskvla/README.md](project/briskvla/README.md#results).

<table>
<thead><tr><th>Model</th><th>Role</th><th>Depth / Params</th><th>VLM depths</th><th>FFN width</th><th>Hidden map</th><th>K</th><th>Spatial</th><th>Object</th><th>Goal</th><th>Long</th><th>Avg</th></tr></thead>
<tbody>
<tr><td rowspan="2"><code>teacher</code></td><td rowspan="2">Teacher</td><td rowspan="2"><code>16L</code> (98.2M)</td><td rowspan="2"><code>0..15</code></td><td rowspan="2">2048</td><td rowspan="2">—</td><td>10</td><td>79</td><td>91</td><td>89</td><td>69</td><td><b>82.0</b></td></tr>
<tr><td>3</td><td>76</td><td>90</td><td>89</td><td>71</td><td><b>81.5</b></td></tr>
<tr><td rowspan="2"><code>student_8L_early</code></td><td rowspan="2">Distilled</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,1,2,3,4,5,6,7</code></td><td rowspan="2">2048</td><td rowspan="2"><code>same_depth</code></td><td>10</td><td>77.3</td><td>86.3</td><td>83.0</td><td>66.3</td><td><b>78.3</b></td></tr>
<tr><td>3</td><td>69.0</td><td>45.7</td><td>52.0</td><td>49.0</td><td>53.9</td></tr>
<tr><td rowspan="2"><code>student_8L_spread</code></td><td rowspan="2">Distilled</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,1,4,5,8,9,12,13</code></td><td rowspan="2">2048</td><td rowspan="2"><code>same_depth</code></td><td>10</td><td>75.3</td><td>82.7</td><td>85.0</td><td>70.0</td><td><b>78.3</b></td></tr>
<tr><td>3</td><td>53.7</td><td>63.7</td><td>69.3</td><td>49.0</td><td>58.9</td></tr>
<tr><td rowspan="2"><code>student_8L_late</code></td><td rowspan="2">Distilled</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>8,9,10,11,12,13,14,15</code></td><td rowspan="2">2048</td><td rowspan="2"><code>same_depth</code></td><td>10</td><td>67</td><td>82</td><td>78</td><td>56</td><td>70.8</td></tr>
<tr><td>3</td><td>53</td><td>66</td><td>69</td><td>45</td><td>58.3</td></tr>
<tr><td rowspan="2"><code>shallow_8L_early</code></td><td rowspan="2">Shallow (GT-only)</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,1,2,3,4,5,6,7</code></td><td rowspan="2">2048</td><td rowspan="2">—</td><td>10</td><td>73.7</td><td>86.3</td><td>74.7</td><td>67.3</td><td>75.5</td></tr>
<tr><td>3</td><td>81.0</td><td>85.3</td><td>75.3</td><td>73.0</td><td><b>78.7</b></td></tr>
<tr><td rowspan="2"><code>shallow_8L_spread</code></td><td rowspan="2">Shallow (GT-only)</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,1,4,5,8,9,12,13</code></td><td rowspan="2">2048</td><td rowspan="2">—</td><td>10</td><td>63.0</td><td>80.3</td><td>79.3</td><td>64.0</td><td>71.7</td></tr>
<tr><td>3</td><td>68.7</td><td>86.0</td><td>77.3</td><td>68.3</td><td>75.1</td></tr>
<tr><td rowspan="2"><code>shallow_8L_late</code></td><td rowspan="2">Shallow (GT-only)</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>8,9,10,11,12,13,14,15</code></td><td rowspan="2">2048</td><td rowspan="2">—</td><td>10</td><td>63</td><td>66</td><td>75</td><td>48</td><td>63.0</td></tr>
<tr><td>3</td><td>56</td><td>71</td><td>77</td><td>47</td><td>62.8</td></tr>
<tr><td rowspan="2"><code>student_8L_early_shallow_pi</code></td><td rowspan="2">Distilled (+GT, no output loss)</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,1,2,3,4,5,6,7</code></td><td rowspan="2">2048</td><td rowspan="2"><code>same_depth</code></td><td>10</td><td>67</td><td>79</td><td>74</td><td>69</td><td>72.2</td></tr>
<tr><td>3</td><td>74</td><td>90</td><td>77</td><td>69</td><td>77.5</td></tr>
<tr><td rowspan="2"><code>student_8L_spread_shallow_pi</code></td><td rowspan="2">Distilled (+GT, no output loss)</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,1,4,5,8,9,12,13</code></td><td rowspan="2">2048</td><td rowspan="2"><code>same_depth</code></td><td>10</td><td>63</td><td>84</td><td>74</td><td>66</td><td>71.8</td></tr>
<tr><td>3</td><td>72</td><td>82</td><td>74</td><td>62</td><td>72.5</td></tr>
<tr><td rowspan="2"><code>student_8L_late_shallow_pi</code></td><td rowspan="2">Distilled (+GT, no output loss)</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>8,9,10,11,12,13,14,15</code></td><td rowspan="2">2048</td><td rowspan="2"><code>same_depth</code></td><td>10</td><td>55</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>3</td><td>56</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td rowspan="2"><code>student_8L_alternating_ffn_fused</code></td><td rowspan="2">Distilled + FFN Fusion</td><td rowspan="2"><code>8L</code> (84.5M)</td><td rowspan="2"><code>0,3,4,7,8,11,12,15</code></td><td rowspan="2">4096</td><td rowspan="2"><code>uniform</code></td><td>10</td><td>76</td><td>83</td><td>85</td><td>61</td><td>76.3</td></tr>
<tr><td>3</td><td>62</td><td>73</td><td>71</td><td>44</td><td>62.5</td></tr>
<tr><td rowspan="2"><code>student_8L_alternating</code></td><td rowspan="2">Distilled (drop control)</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,3,4,7,8,11,12,15</code></td><td rowspan="2">2048</td><td rowspan="2"><code>uniform</code></td><td>10</td><td>70</td><td>81</td><td>82</td><td>52</td><td>71.3</td></tr>
<tr><td>3</td><td>69</td><td>63</td><td>71</td><td>45</td><td>62.0</td></tr>
<tr><td rowspan="2"><code>student_8L_spread</code></td><td rowspan="2">Distilled</td><td rowspan="2"><code>8L</code> (49.1M)</td><td rowspan="2"><code>0,1,4,5,8,9,12,13</code></td><td rowspan="2">2048</td><td rowspan="2"><code>uniform</code></td><td>10</td><td>69</td><td>81</td><td>81</td><td>73</td><td>76.0</td></tr>
<tr><td>3</td><td>51</td><td>72</td><td>66</td><td>42</td><td>57.8</td></tr>
<tr><td><code>shallow_8L_alternating</code></td><td>Shallow (GT-only drop control)</td><td><code>8L</code> (49.1M)</td><td><code>0,3,4,7,8,11,12,15</code></td><td>2048</td><td>—</td><td colspan="6" align="center"><i>not yet trained</i></td></tr>
<tr><td><code>shallow_8L_alternating_ffn_fused</code></td><td>Shallow (GT-only) + FFN Fusion</td><td><code>8L</code> (84.5M)</td><td><code>0,3,4,7,8,11,12,15</code></td><td>4096</td><td>—</td><td colspan="6" align="center"><i>not yet trained</i></td></tr>
<tr><td rowspan="2"><code>shallow_8L_&lt;placement&gt;[_ffn_fused]_untrained</code></td><td rowspan="2">Untrained (zero-shot)</td><td rowspan="2"><code>8L</code></td><td rowspan="2">any above</td><td rowspan="2">2048 / 4096</td><td rowspan="2">—</td><td>10</td><td>0–3</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td>3</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><td><code>student_4L_alternating_ffn_fused</code> / <code>shallow_4L_alternating_ffn_fused</code></td><td>4-layer iterative fusion (<code>16→8→4</code>)</td><td><code>4L</code></td><td><code>0,7,8,15</code></td><td>8192</td><td>—</td><td colspan="6" align="center"><i>supported by <code>--source</code>, not yet trained</i></td></tr>
</tbody>
</table>

`student_8L_late_shallow_pi` was evaluated on `libero_spatial` only.

The 95% confidence interval for a single 100-episode cell is roughly ±8–9 percentage points (about ±4–4.5 points for the 400-episode average).
