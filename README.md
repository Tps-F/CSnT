# CSnT: 生物模倣的自己注意機構を導入したスパイキングニューラルネットワークによる神経動態予測モデルの提案

このリポジトリには本研究にて私が提案したCSnTモデルの実装を公開します。

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
従来の自己注意メカニズム・Attentionは以下の関数で表される。

$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

ここに、スパイクタイミング依存可塑性(STDP)を用いたスパイク活動の変調について、以下のように定義する。

$${S_{ij} = \gamma \sum_{t} \text{spike}_i(t)\text{spike}_j(t)e^{-|t-t'|/\tau}}$$

ここで、

https://github.com/Tps-F/CSnT/blob/149e9b3e108fd41aa8f04232ec6236eebf524491/modules/c2.py#L313

$`\text{spike}_i(t)\text{spike}_j(t)`$ は二つの同期生について相関を表し、
https://github.com/Tps-F/CSnT/blob/ac0777b500fb21446bacde58562368b973551401/modules/c2.py#L276-L283
$`e^{-|t-t'|/\tau}`$ は時間減衰、$`\gamma`$はスケーリングパラメータである。

これらを用いて、生物学的自己注意メカニズムを、

$$ {\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}}(1 + S))V} $$

と定義した。

### Hodgkin-Huxley型ニューロンモデルの応用
以下のような通常のhodgkin-huxley型ニューロンモデルに、適応機能を追加したモデルを作成した。

最初に、膜電位$`V`$の時間発展について

$${C_m \frac{dV}{dt} = -\sum I_{ion} + I_{ext} = -g_{Na}m^3h(V-E_{Na}) - g_K n^4(V-E_K) - g_L(V-E_L) + I_{ext}}$$

と表す。

ゲート変数の動力源$`m ,h, n`$は、
https://github.com/Tps-F/CSnT/blob/149e9b3e108fd41aa8f04232ec6236eebf524491/modules/c2.py#L55-L57

$$ {\frac{dx}{dt} = \alpha_x(V)(1-x) - \beta_x(V)x}$$

と表せる。

$`\alpha_x(V), \beta_x(V)`$といった遷移確率は以下のように表される。
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

