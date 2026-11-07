# refs_verified.md — 参考文献一手来源核验

核验对象：`drafts/paper2_snap_order.md` §9 References（12 条，原表标注"待核"）。
核验方式：arXiv 官方 API（export.arxiv.org/api/query）、Crossref REST API（api.crossref.org）、
Springer 书页（link.springer.com）、NeurIPS 官方 proceedings 索引（proceedings.neurips.cc）、
timm 官方 README（raw.githubusercontent.com）。每条均附一手 URL。
**本文件不含任何推测值**：拿不到的页码/DOI 一律写「未取得」。

核验日期：2026-10-27。

---

## 0. 总判决

| 结论 | 条数 |
|---|---|
| 真实存在、作者/题名/会议/年份与一手来源一致 | 12 / 12 |
| 需补卷期页码/DOI/arXiv 编号（内容本身无错） | 12 / 12 |
| 作者或题名错误 | 0 |
| 会议或年份错配（预印本冒充正式发表） | 0 |
| 不存在 | 0 |

**注意**：题述提到的 Bilinear MLP / I-ViT / I-BERT / SmoothQuant / LLM.int8 在本草稿中
**一次都没有出现**（已对全文 grep：0 命中）。这 5 个方法名不在本文的 12 条参考文献里，
无需核验（可能属于另一篇草稿）。

---

## 1. 逐条核验表

编号沿用草稿 §9 的顺序。

### [1] Bengio / STE
- **草稿写的**：Bengio Y, Léonard N, Courville A. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv, 2013.
- **判决**：**需修正**（仅缺 arXiv 编号；作者、题名、年份全对；属预印本，草稿标 arXiv 正确）
- **一手来源**：https://arxiv.org/abs/1308.3432 （arXiv API 记录：Yoshua Bengio; Nicholas Léonard; Aaron Courville，2013-08-15 提交，无 journal-ref、无 DOI → 确为纯预印本）
- **修正后条目**：
  ```
  [1] Y. Bengio, N. Léonard, and A. Courville, "Estimating or propagating gradients through stochastic neurons for conditional computation," arXiv preprint arXiv:1308.3432, Aug. 2013.
  ```
- **附注**：同年另有一篇单作者预印本 Y. Bengio, "Estimating or propagating gradients through stochastic neurons," arXiv:1305.2982（2013-05-14，单作者）。1308.3432 的 arXiv comment 自述"substantial text overlap with arXiv:1305.2982"。两篇都可作为 STE 出处，但**不要混用作者串**（1305.2982 只有 Bengio 一人）。

### [2] DoReFa-Net
- **草稿写的**：Zhou S, Wu Y, Ni Z, et al. DoReFa-Net: Training low bitwidth convolutional neural networks with low bitwidth gradients. arXiv, 2016.
- **判决**：**需修正**（仅缺 arXiv 编号与完整作者）
- **一手来源**：https://arxiv.org/abs/1606.06160 （arXiv API：Shuchang Zhou; Yuxin Wu; Zekun Ni; Xinyu Zhou; He Wen; Yuheng Zou，2016-06-20，无 journal-ref → 纯预印本）
- **修正后条目**：
  ```
  [2] S. Zhou, Y. Wu, Z. Ni, X. Zhou, H. Wen, and Y. Zou, "DoReFa-Net: Training low bitwidth convolutional neural networks with low bitwidth gradients," arXiv preprint arXiv:1606.06160, Jun. 2016.
  ```

### [3] PACT
- **草稿写的**：Choi J, Wang Z, Venkataramani S, et al. PACT: Parameterized clipping activation for quantized neural networks. arXiv, 2018.
- **判决**：**需修正**（仅缺 arXiv 编号与完整作者）
- **一手来源**：https://arxiv.org/abs/1805.06085 （arXiv API：Jungwook Choi; Zhuo Wang; Swagath Venkataramani; Pierce I-Jen Chuang; Vijayalakshmi Srinivasan; Kailash Gopalakrishnan，2018-05-16，无 journal-ref → 该文从未正式发表于会议/期刊，按预印本引是对的）
- **修正后条目**：
  ```
  [3] J. Choi, Z. Wang, S. Venkataramani, P. I.-J. Chuang, V. Srinivasan, and K. Gopalakrishnan, "PACT: Parameterized clipping activation for quantized neural networks," arXiv preprint arXiv:1805.06085, May 2018.
  ```

### [4] LSQ
- **草稿写的**：Esser S K, McKinstry J L, Bablani D, et al. Learned step size quantization. ICLR, 2020.
- **判决**：**需修正**（缺完整作者与 arXiv 编号；**ICLR 2020 的会议归属正确**，不是预印本冒充）
- **一手来源**：https://arxiv.org/abs/1902.08153 —— arXiv 元数据 comment 字段由作者本人写为 "International Conference on Learning Representations (2020)"，即正式发表于 ICLR 2020；作者 Steven K. Esser; Jeffrey L. McKinstry; Deepika Bablani; Rathinakumar Appuswamy; Dharmendra S. Modha
- **修正后条目**：
  ```
  [4] S. K. Esser, J. L. McKinstry, D. Bablani, R. Appuswamy, and D. S. Modha, "Learned step size quantization," in Proc. Int. Conf. Learn. Represent. (ICLR), 2020. (arXiv:1902.08153)
  ```
- **未取得**：OpenReview 论坛页 URL —— openreview.net 的 API 对本机返回 403 ChallengeRequired，取不到可核验的 forum id，故不写。ICLR 2020 归属以 arXiv 作者元数据为准。

### [5] Polyak & Juditsky
- **草稿写的**：Polyak B T, Juditsky A B. Acceleration of stochastic approximation by averaging. SIAM Journal on Control and Optimization, 1992.
- **判决**：**需修正**（缺卷期页码与 DOI；题名、期刊、年份全对）
- **一手来源**：https://doi.org/10.1137/0330046 （Crossref 记录：SIAM Journal on Control and Optimization, vol. 30, no. 4, pp. 838-855, 1992, type journal-article）
- **修正后条目**：
  ```
  [5] B. T. Polyak and A. B. Juditsky, "Acceleration of stochastic approximation by averaging," SIAM J. Control Optim., vol. 30, no. 4, pp. 838–855, Jul. 1992, doi: 10.1137/0330046.
  ```

### [6] Mean Teacher
- **草稿写的**：Tarvainen A, Valpola H. Mean teachers are better role models. NeurIPS, 2017.
- **判决**：**需修正**（题名被截断——完整题名有副标题；缺卷/页码/arXiv 编号）
- **一手来源**：https://proceedings.neurips.cc/paper_files/paper/2017/hash/68053af2923e00204c3ca7c6a3150cf7-Abstract.html （NeurIPS 官方 proceedings 索引，2017 卷命中，题名完整为 "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results"）；arXiv: https://arxiv.org/abs/1703.01780
- **修正后条目**：
  ```
  [6] A. Tarvainen and H. Valpola, "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results," in Proc. Adv. Neural Inf. Process. Syst. (NIPS), vol. 30, 2017. (arXiv:1703.01780)
  ```
- **页码：未取得**（见 §3）。

### [7] timm
- **草稿写的**：Wightman R. PyTorch Image Models (timm), 2019. (regnety_008 and the EMA conventions used here)
- **判决**：**需修正**（缺 DOI；软件类文献，年份 2019 与 timm 自己给出的引用格式一致，不是错）
- **一手来源**：timm 官方 README 的 BibTeX（https://raw.githubusercontent.com/huggingface/pytorch-image-models/main/README.md）：
  author = Ross Wightman, title = PyTorch Image Models, year = 2019, publisher = GitHub, doi = 10.5281/zenodo.4414861；Zenodo/DataCite 记录：https://doi.org/10.5281/zenodo.4414861
- **修正后条目**：
  ```
  [7] R. Wightman, "PyTorch Image Models," GitHub repository, 2019, doi: 10.5281/zenodo.4414861. [Online]. Available: https://github.com/rwightman/pytorch-image-models
  ```
- **注意**：10.5281/zenodo.4414861 是 Zenodo 的 **concept DOI**（指向最新版）。若审稿要求钉版，DataCite 的该 DOI 当前解析到 "rwightman/pytorch-image-models: v0.8.10dev0 Release"（2023，30 位贡献者）。IEEE 体下用上面的软件引用式即可，不要把 2023 那个版本标题当论文题名。

### [8] RegNet / R(D)
- **草稿写的**：Radosavovic I, Kosaraju R P, Girshick R, et al. Designing network design spaces. CVPR, 2020.
- **判决**：**需修正**（缺页码与 DOI；另见 §2 第 10 条——正文没点名这篇）
- **一手来源**：https://doi.org/10.1109/CVPR42600.2020.01044 （Crossref：2020 IEEE/CVF CVPR, pp. 10425-10433）；arXiv: https://arxiv.org/abs/2003.13678（comment: CVPR 2020）
- **修正后条目**：
  ```
  [8] I. Radosavovic, R. P. Kosaraju, R. Girshick, K. He, and P. Dollár, "Designing network design spaces," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 10425–10433, doi: 10.1109/CVPR42600.2020.01044.
  ```
- **页码口径提示**：IEEE Xplore/Crossref 为 10425–10433；CVF 开放获取 PDF 常被引作 10428–10436。二者并存，**引用时统一用一种**（建议用 DOI 侧页码，可被审稿人按 DOI 复核）。

### [9] Group Normalization
- **草稿写的**：Wu Y, He K. Group normalization. ECCV, 2018.
- **判决**：**需修正**（缺页码与 DOI、缺 LNCS 卷号）
- **一手来源**：https://doi.org/10.1007/978-3-030-01261-8_1 （Crossref：Lecture Notes in Computer Science, pp. 3-19, 2018）；LNCS 卷号 11217 由 Springer 书页 https://link.springer.com/book/10.1007/978-3-030-01261-8 确认（页面原文："Lecture Notes in Computer Science (LNCS, volume 11217)"）
- **修正后条目**：
  ```
  [9] Y. Wu and K. He, "Group normalization," in Proc. Eur. Conf. Comput. Vis. (ECCV), LNCS vol. 11217, 2018, pp. 3–19, doi: 10.1007/978-3-030-01261-8_1.
  ```

### [10] RMSNorm
- **草稿写的**：Zhang B, Sennrich R. Root mean square layer normalization. NeurIPS, 2019.
- **判决**：**需修正**（缺卷号；页码未取得）
- **一手来源**：https://proceedings.neurips.cc/paper_files/paper/2019/hash/1e8a19426224ca89e83cef47f1e7f53b-Abstract.html （2019 卷官方索引命中）；arXiv comment 自述 "NeurIPS 2019"：https://arxiv.org/abs/1910.07467
- **修正后条目**：
  ```
  [10] B. Zhang and R. Sennrich, "Root mean square layer normalization," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019. (arXiv:1910.07467)
  ```
- **页码：未取得**（见 §3）。

### [11] Attention Is All You Need
- **草稿写的**：Vaswani A, et al. Attention is all you need. NeurIPS, 2017.
- **判决**：**需修正**（IEEE 体要求列全 8 位作者；页码未取得）
- **一手来源**：https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html ；作者全名单由 arXiv API 确认（https://arxiv.org/abs/1706.03762）：Ashish Vaswani; Noam Shazeer; Niki Parmar; Jakob Uszkoreit; Llion Jones; Aidan N. Gomez; Lukasz Kaiser; Illia Polosukhin
- **修正后条目**：
  ```
  [11] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin, "Attention is all you need," in Proc. Adv. Neural Inf. Process. Syst. (NIPS), vol. 30, 2017. (arXiv:1706.03762)
  ```
- **陷阱记录**：Crossref 检索该题名会返回一批 2025 年的 `10.65215/*` "posted-content" 条目，那是抢注/镜像 DOI，**不是** NeurIPS 2017 原文。该文在 NeurIPS 无出版社 DOI，不要给这条安一个 DOI。

### [12] CAAE / UTKFace
- **草稿写的**：Zhang Z, Song Y, Qi H. Age progression/regression by conditional adversarial autoencoder. CVPR, 2017. (UTKFace)
- **判决**：**需修正**（缺页码与 DOI；括号里的 UTKFace 归属需改措辞，见 §2 第 11 条）
- **一手来源**：https://doi.org/10.1109/CVPR.2017.463 （Crossref：2017 IEEE CVPR, pp. 4352-4360）；arXiv comment 自述 "Accepted by IEEE CVPR 2017"：https://arxiv.org/abs/1702.08423
- **修正后条目**：
  ```
  [12] Z. Zhang, Y. Song, and H. Qi, "Age progression/regression by conditional adversarial autoencoder," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), 2017, pp. 4352–4360, doi: 10.1109/CVPR.2017.463.
  ```
- **页码陷阱**：该文常被引作 "pp. 2230–2239"（那是另一份页码表）。以 DOI 侧 4352–4360 为准。

---

## 2. 正文断言核查

逐句对照可取得的摘要/全文（arXiv API 的 abstract 字段、ar5iv 全文、官方 README）。

| # | 正文原句（行号） | 核查结果 |
|---|---|---|
| 1 | §2 "Straight-through estimation for quantized networks goes back to Bengio et al."（L83-84） | **成立**。arXiv:1308.3432 摘要原文："we call this the straight-through estimator"。注意它讲的是随机神经元/条件计算，不是量化；"goes back to" 的措辞恰当，不宜写成"该文提出了量化 STE"。 |
| 2 | §2 "DoReFa-Net fixed low-bit weights/activations with a differentiable proxy"（L84-85） | **措辞不准**。arXiv:1606.06160 摘要说的是低比特权重/激活 + **低比特梯度**（反向时对参数梯度做随机量化），机制是直通估计，不是"differentiable proxy"。建议改为："DoReFa-Net quantized weights, activations and gradients with straight-through estimation"。 |
| 3 | §2 "PACT learned activation clipping"（L85） | **成立**。摘要："uses an activation clipping parameter α that is optimized during training"。 |
| 4 | §2 "LSQ learned the quantizer step size"（L85） | **成立**。摘要："a novel means to estimate and scale the task loss gradient at each weight and activation layer's quantizer step size, such that it can be learned in conjunction with other network parameters"。 |
| 5 | §2 "Averaging iterates (Polyak–Juditsky) and its exponential variant are standard"（L91） | **成立**（属通识性归属，题名与内容一致；本文不据此下强断言）。 |
| 6 | §2 "mean-teacher popularized the EMA-shadow formulation in **self-supervised** training"（L92） | **归属错误（领域错）**。Mean Teacher 是 **semi-supervised**，不是 self-supervised：arXiv:1703.01780 完整题名即 "...improve **semi-supervised** deep learning results"，摘要通篇讲 semi-supervised benchmarks（SVHN/CIFAR-10/ImageNet 少标签）。**必须改为 "semi-supervised"**，否则是事实错误。 |
| 7 | §3.1 "shared GroupNorm(8)"（L111） | **成立**。arXiv:1803.08494 摘要："GN divides the channels into groups and computes within each group the mean and variance"。 |
| 8 | §3.1 "RMSNorm avoids mean subtraction"（L116） | **成立**。arXiv:1910.07467 摘要："we hypothesize that re-centering invariance in LayerNorm is dispensable"，RMSNorm 只按 RMS 归一。 |
| 9 | §3.1 "regnety_008 backbone (timm)"（L108） | **成立**（timm 官方 README 确认其为 timm 模型命名；本条是软件引用，无"提出 X"类断言）。 |
| 10 | §3.1 / §2：RegNet 一篇（草稿 [8]） | **正文从未点名**。全文只有 "regnety_008 backbone (timm)" 这一处间接依赖，没有出现 "RegNet" 或 "Designing network design spaces"。[8] 目前是一条"列了但没引"的条目——要么在 §3.1 点名 RegNet，要么删掉该条。 |
| 11 | 草稿 [12] 括号 "(UTKFace)" | **需改措辞**。CAAE 那篇的全文（ar5iv 全文检索）**没有出现 "UTKFace" 字样**（唯一 "utk" 命中是作者邮箱域名 utk.edu）。UTKFace 是该工作随附发布的数据集、社区通行把它引到这篇，但"该文提出了 UTKFace 这个名字"在一手正文里查不到。建议正文写 "the UTKFace dataset released with [12]"，不要把 [12] 说成"提出了 UTKFace 数据集"。 |
| 12 | 全篇：**正文没有任何 [n] 数字引用标记**（已 grep `\[[0-9]+\]`：0 命中） | **投稿阻断项**。12 条参考文献全部靠"点名"引用，与 §9 列表没有编号链接。IEEE 体必须有 [1]–[12] 的正文标记；改稿时把 §2 / §3.1 / §3 的首次点名处接上编号。 |

---

## 3. 无法证实（检索过但拿不到）

| 项 | 检索/尝试 | 结论 |
|---|---|---|
| [6] Mean Teacher 在 NIPS 30 的**页码** | Crossref `query.container-title="Advances in Neural Information Processing Systems 30"` + 题名检索（命中全是 NIPS 35/37/38 的无关条目）；NeurIPS 官方 proceedings 页面本身不印页码；DBLP API 返回 429/202（限流，未取得） | **未取得**。NeurIPS 官方卷不含页码，属正常情况；IEEE 体下用 "vol. 30, 2017 + arXiv 编号" 即可，勿填二手页码。 |
| [10] RMSNorm 在 NeurIPS 32 的**页码** | 同上（Crossref 无该卷、proceedings 无页码、DBLP 限流） | **未取得**。 |
| [11] Attention 在 NIPS 30 的**页码** | 同上 | **未取得**。常见二手值（5998–6008 一类）**未经一手来源核实，不要直接粘贴**。 |
| [11] Attention 的**出版社 DOI** | Crossref 题名检索只返回 2025 年 `10.65215/*` 抢注 DOI | **不存在**（NeurIPS 2017 无 DOI）。**不要给这条编 DOI**。 |
| [4] LSQ 的 **OpenReview forum URL** | api.openreview.net / api2.openreview.net 均返回 403 ChallengeRequiredError | **未取得**。ICLR 2020 归属改由 arXiv 作者元数据 comment 字段佐证。 |
| [9] GN 的 LNCS 卷号 | Crossref 书记录无 volume 字段 → 改查 Springer 书页 | **已取得**：LNCS vol. 11217（Springer 书页原文），故表中已填。 |
| UTKFace 数据集名称是否出自 [12] 正文 | ar5iv 全文（https://ar5iv.labs.arxiv.org/html/1702.08423）关键词检索 "UTKFace"/"FG-NET"/"IMDB" | **未证实**：正文未出现数据集名。见 §2 第 11 条。 |

---

## 4. 投稿前 checklist（按优先级）

1. **接上 [n] 引用标记**（当前 0 个）——最高优先级。
2. 把 §2 的 "self-supervised" 改成 "semi-supervised"（事实错误）。
3. 按上表替换 §9 全部 12 条（都提供了可直接粘贴的条目）。
4. §2 的 DoReFa 措辞改为 straight-through estimation。
5. 决定 RegNet [8] 是保留（正文点名）还是删除。
6. [12] 的 UTKFace 归属改成 "dataset released with [12]"。
7. 三条 NeurIPS 文献若期刊模板强制要页码，走 IEEE 的 "electronic proceedings" 变体（不填页码、给 proceedings URL），**不要填未核实的二手页码**。
