# CSnT: 生物模倣的自己注意機構を導入したスパイキングニューラルネットワークによる神経動態予測モデルの提案

このリポジトリには本研究にて私が提案したCSnTモデルの実装を公開します。

Repository: https://github.com/Tps-F/CSnT/
## 本研究について
本研究は、ニューロンモデルにおいて生物学的な自己注意機構を導入することで、スパイキングニューラルネットワークの神経動態予測精度を向上させると共に、より生物学的な挙動を再現することを目的としている。
私は昨年までの研究で、脳波を用いた感情分析を行っていた。、その際に$`\theta`$帯の脳波について時系列による変化が激しく、これを前の状態からのシナプス可塑性にによるものだと予想した。
それを踏まえて、SNNに自己注意メカニズムを導入することにより時空間パターンの学習がより容易になり、精度向上や生物学的な再現度の向上につながると考え、本研究を行った。

結果として、単純なSNNレイヤーに通常の自己注意機構を導入したときのスパイクの予測精度は48%であるのに対し、提案手法を導入したCSnTモデルでは79%まで向上させることができた。

## モデルのアーキテクチャについて
モデルの構造は以下のように作成した。

![Architecture](https://github.com/user-attachments/assets/44058c2d-df56-4e63-84c0-f37674adba4f "Architecture of CSnT")
構造として、Biological SNN Layer, Neural Transformer, Loss Computationに分かれる。
以下に、新たに提唱及び改変したモジュールについて説明する。
### 生物学的自己注意メカニズムの提案
まず最初に、自己注意メカニズムに時間的な相関を捉えられるようなモジュールの提案をした。
最初に、スパイクについて、

https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L304-L315

$$ A_{spike} = \frac{SS^T}{\sqrt{d}} $$

と表し、スパイクの時空間的な特徴を捉える。

ここで、時間減衰を
https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L280-L282

$$ K(\Delta t) = \exp(-|\Delta t|/\tau) $$

と表すことで、自己注意メカニズムを

https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L325-L333

$$ A(Q,K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot K(\Delta t) \odot (1 + A_{spike})\right)V $$

$$ \text{Context} = A(Q,K)V \odot \sigma(g(V_m)) $$

と定義した。

### SNNレイヤーの作成

#### Synaptic Plasticity(STDP)の実装
実際のシナプスの学習メカニズムを模倣することを目的に、STDP を導入した。
プレシナプスとポストシナプスの動力学について

$$ \frac{dx_{pre}}{dt} = -\frac{x_{pre}}{\tau_{pre}} + \sum_f \delta(t - t_f^{pre})$$
$$\frac{dx_{post}}{dt} = -\frac{x_{post}}{\tau_{post}} + \sum_f \delta(t - t_f^{post})$$

と定める。
これらについて、重みの変更を行なっていく。
https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L107-L120

$$ \frac{dw}{dt} = \eta(x_{post}(t)S_{pre}(t) - x_{pre}(t)S_{post}(t))$$

出力は、$`x_{pre}`$とひとつ前の重みとのの全層結合を返す。それを膜電位$`V`$としてHodgkin-Huxley型ニューロンモデルへと渡す。

https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L104-L105
https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L231-L237

#### Hodgkin-Huxley型ニューロンモデルの応用
以下のような通常のhodgkin-huxley型ニューロンモデルに、適応機能を追加したモデルを作成した。

最初に、膜電位$`V`$の時間発展について

$$C_m \frac{dV}{dt} = -\sum I_{ion} + I_{ext} = = -(I_{Na} + I_K + I_L) + I_{ext} $$

と表す。

ゲート変数の動力源$`m ,h, n`$は、
https://github.com/Tps-F/CSnT/blob/149e9b3e108fd41aa8f04232ec6236eebf524491/modules/c2.py#L55-L57

$$\frac{dm}{dt} = \alpha_m(V)(1-m) - \beta_m(V)m$$
$$\frac{dh}{dt} = \alpha_h(V)(1-h) - \beta_h(V)h$$
$$\frac{dn}{dt} = \alpha_n(V)(1-n) - \beta_n(V)n$$

と表せる。

また、$`\alpha_x(V), \beta_x(V)`$といった遷移確率は以下のように表される。
https://github.com/Tps-F/CSnT/blob/149e9b3e108fd41aa8f04232ec6236eebf524491/modules/c2.py#L25-L41

$$\begin{split}
\begin{array}{ll}
\alpha_{m}(V)=\dfrac {0.1(25-V)}{\exp \left[(25-V)/10\right]-1}, &\beta_{m}(V)=4\exp {(-V/18)}\\
\alpha_{h}(V)=0.07\exp {(-V/20)}, & \beta_{h}(V)={\dfrac{1}{\exp {\left[(30-V)/10 \right]}+1}}\\
\alpha_{n}(V)={\dfrac {0.01(10-V)}{\exp {\left[(10-V)/10\right]}-1}},& \beta_{n}(V)=0.125\exp {(-V/80)} 
\end{array}
\end{split}
$$

これらを用いて、各イオン電流は以下の形式で表せる。

https://github.com/Tps-F/CSnT/blob/149e9b3e108fd41aa8f04232ec6236eebf524491/modules/c2.py#L64-L66

$$ {I_{Na} = g_{Na}m^3h(V - E_{Na})}$$
$${I_K = g_K n^4(V - E_K)}$$
$${I_L = g_L(V - E_L)}$$

#### LIF Neuronの実装

Hodgkin-Huxley型ニューロンモデルにて得た電流$`I`$と、STDPにて得られた電圧$`V`$を用いて、ニューロンモデルを動かす。

膜電位の動力学を適応変数$`w`$を用いて
https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L163-L165

$$ {\tau_m\frac{dV}{dt} = -(V-V_{rest}) - w + RI} $$

と表す。$`w`$は
https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L169-L173

$$ {\tau_w\frac{dw}{dt} = a(V-V_{rest}) - w + b\sum_i \delta(t-t_i)} $$

といった時間発展で更新されていく。これによって発火頻度を最適化しているする。

そして、スパイクの生成は
https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L176

$$
\text{if } V \geq V_{thresh}: \begin{cases}
V \rightarrow V_{reset} \\
\text{emit spike}\end{cases}
$$

という階段関数を用いて表した。
