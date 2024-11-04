# CSnT: 生物模倣的自己注意機構を導入したスパイキングニューラルネットワークによる神経動態予測モデルの提案

以下のリポジトリに本研究で提案したCSnTモデルの実装を公開しているので、確認してください。
Repository: https://github.com/Tps-F/CSnT/
## 本研究について
本研究は、ニューロンモデルにおいて生物学的な自己注意機構を導入することで、スパイキングニューラルネットワークの神経動態予測精度を向上させると共に、より生物学的な挙動を再現することを目的としている。
私は昨年までの研究で、脳波を用いた感情分析を行っていた。、その際に$`\theta`$帯の脳波について時系列による変化が激しく、これを前の状態からのシナプス可塑性にによるものだと予想した。
それを踏まえて、SNNに自己注意メカニズムを導入することにより時空間パターンの学習がより容易になり、精度向上や生物学的な再現度の向上につながると考え、本研究を行った。

結果として、単純なSNNレイヤーに通常の自己注意機構を導入したときのスパイクの予測精度は48%であるのに対し、提案手法を導入したCSnTモデルでは79%まで向上させることができた。

## 実行方法
<details>
<summary>実行方法について</summary>
1. uvをインストール
  
```sh
# macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```ps
# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. データセットのダウンロード

```sh
curl -L -o ~/Downloads/archive.zip\
https://www.kaggle.com/api/v1/datasets/download/selfishgene/single-neurons-as-deep-nets-nmda-test-data
```
もしくは、[こちらから](https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data/data)ダウンロード

https://www.kaggle.com/datasets/selfishgene/single-neurons-as-deep-nets-nmda-test-data/data


また、プロジェクトをクローン
```sh
git clone https://github.com/Tps-F/CSnT.git
```

その後、`CSnT/config/config.yaml`中の`nmda_dataset.data_dir`をダウンロードしたデータセットの場所に変更

3. ライブラリのインストール

```sh
uv sync

```

4. トレーニングの開始

```sh
uv run train.py
```

</details>

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

$$ K(\Delta t) = \exp\left(\frac{-|\Delta t|}{\tau}\right) $$

と表すことで、自己注意メカニズムを

https://github.com/Tps-F/CSnT/blob/eeb88a6bdb25ca1d61f3a9fcf2b54621d8ada26a/modules/c2.py#L325-L333

$$ A(Q,K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} \odot K(\Delta t) \odot (1 + A_{spike})\right)V $$

$$ \text{Context} = A(Q,K)V \odot \sigma(g(V_m)) $$

と定義した。
これにより時間マスクを加えることによって、シナプスの電位が時間と共に減衰していくことを再現した。

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


## 現在の結果
<p>
<img src="https://github.com/user-attachments/assets/1cf16336-e74d-44d2-94a5-55acf5eaa12a" width="45%">  <img src="https://github.com/user-attachments/assets/3dde8cf9-29a0-4c1d-8110-8fa55591dcf1" width="45%"></p>
<p>
<img src="https://github.com/user-attachments/assets/ac5b70ca-e369-40f2-8595-b8f41f8002c2" width="45%">  <img src="https://github.com/user-attachments/assets/31074f71-a157-4711-8f91-8203d16b9171" width="45%"></p>

このように、DVT Loss は綺麗に下がっていっている一方、Spike のLoss は一度下がってから安定していな
いことがわかる。Spike について、正答率は最大0.79、Loss は最小0.0045 に達した。

また、作成したモデルを用いて推論した神経活動について下に記載する。
<img src="https://github.com/user-attachments/assets/b6b9e8e0-66b3-42ea-904c-7c9f4242f083" width="80%">


## 考察・今後の展望
結果でも述べたとおり、Spike のLoss が一度下がってから安定しないのが問題である。現在考えている原因として、プレシナプスとポストシナプスの初期値を固定しているので学習結果が反映されていない、データセットの事前処理について正規化などを行なっていないなどさまざまな原因が考えられるので、今後実験を重ねるとともに、正答率90% 超えを目指して今後も開発を続行していこうと思う。

## 結論
今回提案したネットワークは生物学的な動きを再現することで、まだ完成とは言えないがニューロンとして
振る舞えていると結論づけた。しかし、学習の遷移を観察しているとまだ実用には至らない点や、不確定要素
が多すぎることを今後改善していくことで、一つのニューロンモデルとして再現しようと考える。

## 参考文献
[1] Hodgkin, A. L., & Huxley, A. F. (1952). A quantitative description of membrane cur- rent and its application to conduction and excitation in nerve. Journal of Physiology, 117(4), 500‒ 544. https://doi.org/10.1113/jphysiol.1952.sp004764 

[2] Bi, G. Q., & Poo, M. M. (1998). Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synap-tic strength, and postsynaptic cell type. Journal of Neuroscience, 18(24), 10464‒ 10472.https://doi.org/10.1523/JNEUROSCI.18-24-10464.1998 

[3] Song, S., Miller, K. D., & Abbott, L. F. (2000). Competitive Hebbian learning through spike timing-dependent plasticity. Nature Neu-roscience, 3(9), 919-926. https://doi.org/10.1038/78829 

[4] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All YouNeed. arXiv preprint arXiv:1706.03762. https://doi.org/10.48550/arXiv.1706.03762 

[5] London, M., & Häusser, M. (2005). Dendritic computation. Annual Review of Neuroscience, 28, 503‒532. https://doi.org/10.1146/annurev.neuro.28.061604.135703 

[6] Bai, S., Kolter, J. Z., & Koltun,V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequencemodeling. arXiv preprint arXiv:1803.01271. https://doi.org/10.48550/arXiv.1803.01271 

[7] Yu, F.,& Koltun, V. (2016). Multi-scale context aggregation by dilated convolutions. arXiv preprintarXiv:1511.07122v3. https://doi.org/10.48550/arXiv.1511.07122 

[8] Neftci, E. O., Mostafa, H., & Zenke, F. (2019). Surrogate gradient learning in spiking neural networks. arXiv preprint arXiv:1901.09948v2.https://doi.org/10.48550/arXiv.1901.09948

[9] David Beniaguev, Idan Segev and Michael London. "Single cortical neurons as deep artificial neural networks." Neuron. 2021; 109: 2727-2739.e3 doi:https://doi.org/10.1016/j.neuron.2021.07.002

[10] Hay, Etay, Sean Hill, Felix Schürmann, Henry Markram, and Idan Segev. 2011. “Models of Neocortical Layer 5b Pyramidal Cells Capturing a Wide Range of Dendritic and Perisomatic Active Properties.” Edited by Lyle J. Graham. PLoS Computational Biology 7 (7): e1002107. doi: https://doi.org/10.1371/journal.pcbi.1002107.
