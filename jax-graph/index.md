---
title: How to Visualise Computational Graph of a JAX Program
lang: en
math: |-
  <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.1/viz.js" integrity="sha384-aD1MJYb0WKIUT+CtwJp5LTuV3U4pLAS6B/nUxL7ECimC2pN9N8vjlMr/yQCAkzxE" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.1/full.render.js" integrity="sha384-bAixY275aIpCj6Te19y0MILZ4V+VEC8CVFujFEH+Lf7W+4XYYeYLwW5IBI6yQmMT" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.0/dist/svg-pan-zoom.min.js" integrity="sha384-3008WpYB2pOBvE7lwkrKf+qTmbTPGGPYxA9C1YVhvbPukns4ZFj7E98QPLkNW9dS" crossorigin="anonymous"></script>
  <script src="https://cdn.jsdelivr.net/npm/@hpcc-js/wasm/dist/index.min.js" integrity="sha384-X+8WXyWZ+W2gUHiSSj0aePAkE77Fl6eZ+QIByw+Ii8LzWEJ/W8bI8M4RkneDAJ4D" crossorigin="anonymous"></script>
---

> Reference: [Jax: Visualising the computational graph of a jax program](http://www.bnikolic.co.uk/blog/python/jax/2020/10/20/jax-outputgraph.html)

Run the following [Python script](https://github.com/ayaka14732/tpu-starter/blob/e9a8c0c/00-basics/test_hlo.py) to generate a compuational graph:

```python
import jax
import jax.lax as lax
from jax.lib import xla_bridge
import jax.numpy as np
from jaxlib.xla_extension import HloPrintOptions

# jax.config.update('jax_platform_name', 'cpu')

backend = xla_bridge.get_backend()
print(backend.platform_version)

option = HloPrintOptions()
# option.print_metadata = False
# option.include_layout_in_shapes = False
# option.print_extra_attributes = False

@jax.xla_computation
def f(x, y):
    a = np.einsum('pqrs,tuqvr->pstuv', x, y)
    return lax.sin(a)

c = f(np.ones((3,4,5,6)), np.ones((7,8,4,9,5)))
module = backend.compile(c).hlo_modules()[0]
hlo_text = module.to_string(option)
print(hlo_text)

with open('/tmp/hlo.txt', 'w') as f:
    print(hlo_text, file=f)
```

Install Bazel build tool:

```sh
sudo mkdir -p /usr/local/lib/bazel/bin
sudo wget -P /usr/local/lib/bazel/bin https://github.com/bazelbuild/bazel/releases/download/5.0.0/bazel-5.0.0-linux-x86_64
sudo chmod +x /usr/local/lib/bazel/bin/bazel-5.0.0-linux-x86_64
```

Compile `interactive_graphviz`:

```sh
git clone --depth=1 https://github.com/tensorflow/tensorflow.git
cd tensorflow
bazel build tensorflow/compiler/xla/tools:interactive_graphviz
```

The compilation takes about 7 minutes on a Cloud TPU VM (with 96 CPUs).

Run `interactive_graphviz`:

```sh
bazel-bin/tensorflow/compiler/xla/tools/interactive_graphviz --hlo_text=/tmp/hlo.txt
```

Input `list computations`:

```
command: list computations
Entry computation:
  xla_computation_f.13

Subcomputations:
  fused_computation.1.clone.clone
  fused_computation.4
  fused_computation.3
```

Input the entry computation according to the output. In this example, it is `xla_computation_f.13`:

```
command: xla_computation_f.13
file:///tmp/interactive_graphviz.1645348689190091.html
```

Open `file:///tmp/interactive_graphviz.1645348689190091.html` to view the computational graph:

<div id="container"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="100%" height="100%" viewBox="0.00 0.00 837.84 956.64" id="graph"><g id="graph0" class="graph" transform="scale(1 1) rotate(0) translate(4 952.6377)"><title>G</title><g id="a_graph0"><a xlink:title=" "><polygon fill="#ffffff" stroke="transparent" points="-4,4 -4,-952.6377 833.8417,-952.6377 833.8417,4 -4,4"></polygon><text text-anchor="start" x="314.9853" y="-919.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">Computation xla_computation_f.14</text></a></g><g id="clust1" class="cluster"><title>cluster_94133549829168</title><g id="a_clust1"><a xlink:title=" "><path fill="#f5f5f5" stroke="#c2c2c2" stroke-width="2" d="M20,-246.6377C20,-246.6377 580,-246.6377 580,-246.6377 586,-246.6377 592,-252.6377 592,-258.6377 592,-258.6377 592,-856.6377 592,-856.6377 592,-862.6377 586,-868.6377 580,-868.6377 580,-868.6377 20,-868.6377 20,-868.6377 14,-868.6377 8,-862.6377 8,-856.6377 8,-856.6377 8,-258.6377 8,-258.6377 8,-252.6377 14,-246.6377 20,-246.6377"></path><text text-anchor="start" x="217.1858" y="-853.0377" font-family="Times,serif" font-size="14.00" fill="#000000">Fused expression for </text><text text-anchor="start" x="337.3184" y="-853.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">fusion.6</text><text text-anchor="start" x="262.8615" y="-839.0377" font-family="Times,serif" font-size="14.00" fill="#000000">output fusion</text><text text-anchor="start" x="260.6635" y="-825.0377" font-family="Times,serif" font-size="14.00" fill="#000000">kind=kOutput</text></a></g></g><g id="clust2" class="cluster"><title>cluster_94133549829376</title><g id="a_clust2"><a xlink:title=" "><path fill="#f5f5f5" stroke="#c2c2c2" stroke-width="2" d="M316,-490.6377C316,-490.6377 572,-490.6377 572,-490.6377 578,-490.6377 584,-496.6377 584,-502.6377 584,-502.6377 584,-652.6377 584,-652.6377 584,-658.6377 578,-664.6377 572,-664.6377 572,-664.6377 316,-664.6377 316,-664.6377 310,-664.6377 304,-658.6377 304,-652.6377 304,-652.6377 304,-502.6377 304,-502.6377 304,-496.6377 310,-490.6377 316,-490.6377"></path><text text-anchor="start" x="361.1858" y="-649.0377" font-family="Times,serif" font-size="14.00" fill="#000000">Fused expression for </text><text text-anchor="start" x="481.3184" y="-649.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">fusion.9</text><text text-anchor="start" x="412.3068" y="-635.0377" font-family="Times,serif" font-size="14.00" fill="#000000">loop fusion</text><text text-anchor="start" x="409.3311" y="-621.0377" font-family="Times,serif" font-size="14.00" fill="#000000">kind=kLoop</text></a></g></g><g id="clust3" class="cluster"><title>cluster_94133549829584</title><g id="a_clust3"><a xlink:title=" "><path fill="#f5f5f5" stroke="#c2c2c2" stroke-width="2" d="M28,-490.6377C28,-490.6377 284,-490.6377 284,-490.6377 290,-490.6377 296,-496.6377 296,-502.6377 296,-502.6377 296,-754.6377 296,-754.6377 296,-760.6377 290,-766.6377 284,-766.6377 284,-766.6377 28,-766.6377 28,-766.6377 22,-766.6377 16,-760.6377 16,-754.6377 16,-754.6377 16,-502.6377 16,-502.6377 16,-496.6377 22,-490.6377 28,-490.6377"></path><text text-anchor="start" x="73.1858" y="-751.0377" font-family="Times,serif" font-size="14.00" fill="#000000">Fused expression for </text><text text-anchor="start" x="193.3184" y="-751.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">fusion.8</text><text text-anchor="start" x="124.3068" y="-737.0377" font-family="Times,serif" font-size="14.00" fill="#000000">loop fusion</text><text text-anchor="start" x="121.3311" y="-723.0377" font-family="Times,serif" font-size="14.00" fill="#000000">kind=kLoop</text></a></g></g><!-- 94133550737152 --><g id="node1" class="node"><title>94133550737152</title><polygon fill="#ffb74d" stroke="#c88719" points="260.1836,-912.6377 51.8164,-912.6377 51.8164,-876.6377 260.1836,-876.6377 260.1836,-912.6377"></polygon><text text-anchor="start" x="122.3783" y="-897.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">Parameter 1</text><text text-anchor="start" x="59.6583" y="-883.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[7,8,4,9,5]{3,2,4,1,0:T(4,128)}</text></g><!-- 94133550110848 --><g id="node7" class="node"><title>94133550110848</title><polygon fill="#ffb74d" stroke="#c88719" points="260.1836,-810.6377 51.8164,-810.6377 51.8164,-774.6377 260.1836,-774.6377 260.1836,-810.6377"></polygon><text text-anchor="start" x="122.3783" y="-795.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">Parameter 0</text><text text-anchor="start" x="59.6583" y="-781.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[7,8,4,9,5]{3,2,4,1,0:T(4,128)}</text></g><!-- 94133550737152&#45;&gt;94133550110848 --><g id="edge5" class="edge"><title>94133550737152-&gt;94133550110848</title><g id="a_edge5"><a xlink:title="parameter.2 -> param_0.16"><path fill="none" stroke="#000000" d="M156,-876.4021C156,-860.9547 156,-838.5526 156,-820.8017"></path><polygon fill="#000000" stroke="#000000" points="159.5001,-820.7153 156,-810.7154 152.5001,-820.7154 159.5001,-820.7153"></polygon></a></g></g><!-- 94133550738048 --><g id="node2" class="node"><title>94133550738048</title><polygon fill="#ffe0b2" stroke="#cbae82" points="808.6837,-912.6377 621.3163,-912.6377 621.3163,-876.6377 808.6837,-876.6377 808.6837,-912.6377"></polygon><text text-anchor="start" x="681.3783" y="-897.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">Parameter 0</text><text text-anchor="start" x="629.1583" y="-883.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[3,4,5,6]{3,1,2,0:T(4,128)}</text></g><!-- 94133550738944 --><g id="node3" class="node"><title>94133550738944</title><polygon fill="#ffffff" stroke="#000000" points="829.6836,-810.6377 600.3164,-810.6377 600.3164,-774.6377 829.6836,-774.6377 829.6836,-810.6377"></polygon><text text-anchor="start" x="696.7272" y="-795.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">bitcast</text><text text-anchor="start" x="608.1583" y="-781.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[3,4,5,6,1,1]{3,1,5,4,2,0:T(4,128)}</text></g><!-- 94133550738048&#45;&gt;94133550738944 --><g id="edge1" class="edge"><title>94133550738048-&gt;94133550738944</title><g id="a_edge1"><a xlink:title="parameter.1 -> bitcast"><path fill="none" stroke="#000000" d="M715,-876.4021C715,-860.9547 715,-838.5526 715,-820.8017"></path><polygon fill="none" stroke="#000000" points="718.5001,-820.7153 715,-810.7154 711.5001,-820.7154 718.5001,-820.7153"></polygon></a></g></g><!-- 94133550109056 --><g id="node4" class="node"><title>94133550109056</title><polygon fill="#ffe0b2" stroke="#cbae82" points="574.6836,-708.6377 345.3164,-708.6377 345.3164,-672.6377 574.6836,-672.6377 574.6836,-708.6377"></polygon><text text-anchor="start" x="426.3783" y="-693.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">Parameter 1</text><text text-anchor="start" x="353.1583" y="-679.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[3,4,5,6,1,1]{3,1,5,4,2,0:T(4,128)}</text></g><!-- 94133550738944&#45;&gt;94133550109056 --><g id="edge2" class="edge"><title>94133550738944-&gt;94133550109056</title><g id="a_edge2"><a xlink:title="bitcast -> param_1.9"><path fill="none" stroke="#000000" d="M669.9981,-774.6369C626.6586,-757.3011 561.0321,-731.0505 514.6979,-712.5168"></path><polygon fill="none" stroke="#000000" points="515.9944,-709.2659 505.4098,-708.8016 513.3946,-715.7652 515.9944,-709.2659"></polygon></a></g></g><!-- 94133550107264 --><g id="node5" class="node"><title>94133550107264</title><polygon fill="#ffe0b2" stroke="#cbae82" points="566.6836,-606.6377 337.3164,-606.6377 337.3164,-570.6377 566.6836,-570.6377 566.6836,-606.6377"></polygon><text text-anchor="start" x="418.3783" y="-591.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">Parameter 0</text><text text-anchor="start" x="345.1583" y="-577.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[3,4,5,6,1,1]{3,1,5,4,2,0:T(4,128)}</text></g><!-- 94133550109056&#45;&gt;94133550107264 --><g id="edge3" class="edge"><title>94133550109056-&gt;94133550107264</title><g id="a_edge3"><a xlink:title="param_1.9 -> param_0.17"><path fill="none" stroke="#000000" d="M458.5698,-672.4021C457.3582,-656.9547 455.6012,-634.5526 454.2089,-616.8017"></path><polygon fill="none" stroke="#000000" points="457.6891,-616.4111 453.4179,-606.7154 450.7106,-616.9585 457.6891,-616.4111"></polygon></a></g></g><!-- 94133550108160 --><g id="node6" class="node"><title>94133550108160</title><polygon fill="#c8e6c9" stroke="#97b498" points="575.5051,-534.6377 312.4949,-534.6377 312.4949,-498.6377 575.5051,-498.6377 575.5051,-534.6377"></polygon><text text-anchor="start" x="421.6427" y="-519.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">copy.19</text><text text-anchor="start" x="320.2477" y="-505.0377" font-family="Times,serif" font-size="14.00" fill="#000000">bf16[3,4,5,6,1,1]{3,2,1,5,4,0:T(8,128)(2,1)}</text></g><!-- 94133550107264&#45;&gt;94133550108160 --><g id="edge4" class="edge"><title>94133550107264-&gt;94133550108160</title><g id="a_edge4"><a xlink:title="param_0.17 -> copy.19"><path fill="none" stroke="#000000" d="M449.9813,-570.4691C449.1257,-562.7687 448.1083,-553.612 447.1574,-545.0543"></path><polygon fill="none" stroke="#000000" points="450.6289,-544.6033 446.0459,-535.051 443.6717,-545.3763 450.6289,-544.6033"></polygon></a></g></g><!-- 94133550498816 --><g id="node11" class="node"><title>94133550498816</title><polygon fill="#1565c0" stroke="#003c8f" points="513.0041,-462.6377 86.9959,-462.6377 86.9959,-398.6377 513.0041,-398.6377 513.0041,-462.6377"></polygon><text text-anchor="start" x="261.3068" y="-447.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#ffffff">convolution.5</text><text text-anchor="start" x="94.7481" y="-433.0377" font-family="Times,serif" font-size="14.00" fill="#ffffff">window={size=1x7x8x4 pad=0_0x6_6x7_7x0_0 rhs_reversal=0x1x1x0}</text><text text-anchor="start" x="196.4511" y="-419.0377" font-family="Times,serif" font-size="14.00" fill="#ffffff">dim_labels=03fb12_123oi0-&gt;0b12f3</text><text text-anchor="start" x="193.1583" y="-405.0377" font-family="Times,serif" font-size="14.00" fill="#ffffff">f32[3,6,7,8,9,1]{4,1,5,3,2,0:T(8,128)}</text></g><!-- 94133550108160&#45;&gt;94133550498816 --><g id="edge9" class="edge"><title>94133550108160-&gt;94133550498816</title><g id="a_edge9"><a xlink:title="copy.19 -> convolution.5"><path fill="none" stroke="#000000" d="M413.8129,-498.6093C398.8585,-489.6782 380.2213,-478.5476 362.3855,-467.8957"></path><polygon fill="none" stroke="#000000" points="364.0939,-464.8394 353.7139,-462.7168 360.5047,-470.8492 364.0939,-464.8394"></polygon></a></g><text text-anchor="middle" x="373.6098" y="-460.5541" font-family="Times,serif" font-size="14.00" fill="#000000">0</text></g><!-- 94133550104576 --><g id="node8" class="node"><title>94133550104576</title><polygon fill="#ffb74d" stroke="#c88719" points="260.1836,-708.6377 51.8164,-708.6377 51.8164,-672.6377 260.1836,-672.6377 260.1836,-708.6377"></polygon><text text-anchor="start" x="122.3783" y="-693.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">Parameter 0</text><text text-anchor="start" x="59.6583" y="-679.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[7,8,4,9,5]{3,2,4,1,0:T(4,128)}</text></g><!-- 94133550110848&#45;&gt;94133550104576 --><g id="edge6" class="edge"><title>94133550110848-&gt;94133550104576</title><g id="a_edge6"><a xlink:title="param_0.16 -> param_0.15"><path fill="none" stroke="#000000" d="M156,-774.4021C156,-758.9547 156,-736.5526 156,-718.8017"></path><polygon fill="#000000" stroke="#000000" points="159.5001,-718.7153 156,-708.7154 152.5001,-718.7154 159.5001,-718.7153"></polygon></a></g></g><!-- 94133550105472 --><g id="node9" class="node"><title>94133550105472</title><polygon fill="#c8e6c9" stroke="#97b498" points="277.0051,-606.6377 34.9949,-606.6377 34.9949,-570.6377 277.0051,-570.6377 277.0051,-606.6377"></polygon><text text-anchor="start" x="133.6427" y="-591.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">copy.17</text><text text-anchor="start" x="42.7477" y="-577.0377" font-family="Times,serif" font-size="14.00" fill="#000000">bf16[7,8,4,9,5]{3,4,2,1,0:T(8,128)(2,1)}</text></g><!-- 94133550104576&#45;&gt;94133550105472 --><g id="edge7" class="edge"><title>94133550104576-&gt;94133550105472</title><g id="a_edge7"><a xlink:title="param_0.15 -> copy.17"><path fill="none" stroke="#000000" d="M156,-672.4021C156,-656.9547 156,-634.5526 156,-616.8017"></path><polygon fill="#000000" stroke="#000000" points="159.5001,-616.7153 156,-606.7154 152.5001,-616.7154 159.5001,-616.7153"></polygon></a></g></g><!-- 94133550106368 --><g id="node10" class="node"><title>94133550106368</title><polygon fill="#ffffff" stroke="#000000" points="287.5051,-534.6377 24.4949,-534.6377 24.4949,-498.6377 287.5051,-498.6377 287.5051,-534.6377"></polygon><text text-anchor="start" x="132.4772" y="-519.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">bitcast.9</text><text text-anchor="start" x="32.2477" y="-505.0377" font-family="Times,serif" font-size="14.00" fill="#000000">bf16[7,8,4,9,5,1]{3,4,2,1,0,5:T(8,128)(2,1)}</text></g><!-- 94133550105472&#45;&gt;94133550106368 --><g id="edge8" class="edge"><title>94133550105472-&gt;94133550106368</title><g id="a_edge8"><a xlink:title="copy.17 -> bitcast.9"><path fill="none" stroke="#000000" d="M156,-570.4691C156,-562.7687 156,-553.612 156,-545.0543"></path><polygon fill="#000000" stroke="#000000" points="159.5001,-545.0509 156,-535.051 152.5001,-545.051 159.5001,-545.0509"></polygon></a></g></g><!-- 94133550106368&#45;&gt;94133550498816 --><g id="edge10" class="edge"><title>94133550106368-&gt;94133550498816</title><g id="a_edge10"><a xlink:title="bitcast.9 -> convolution.5"><path fill="none" stroke="#000000" d="M186.1871,-498.6093C201.1415,-489.6782 219.7787,-478.5476 237.6145,-467.8957"></path><polygon fill="#000000" stroke="#000000" points="239.4953,-470.8492 246.2861,-462.7168 235.9061,-464.8394 239.4953,-470.8492"></polygon></a></g><text text-anchor="middle" x="235.0579" y="-475.0676" font-family="Times,serif" font-size="14.00" fill="#000000">1</text></g><!-- 94133550735360 --><g id="node12" class="node"><title>94133550735360</title><g id="a_node12"><a xlink:title="xla_computation(f)/sinop_type: sinsource: test_hlo.py:20"><polygon fill="#fff9c4" stroke="#cbc693" points="414.6836,-362.6377 185.3164,-362.6377 185.3164,-326.6377 414.6836,-326.6377 414.6836,-362.6377"></polygon><text text-anchor="start" x="283.4751" y="-347.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">sine.3</text><text text-anchor="start" x="193.1583" y="-333.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[3,6,7,8,9,1]{4,1,5,3,2,0:T(8,128)}</text></a></g></g><!-- 94133550498816&#45;&gt;94133550735360 --><g id="edge11" class="edge"><title>94133550498816-&gt;94133550735360</title><g id="a_edge11"><a xlink:title="convolution.5 -> sine.3"><path fill="none" stroke="#000000" d="M300,-398.6145C300,-390.2467 300,-381.2847 300,-373.1151"></path><polygon fill="#000000" stroke="#000000" points="303.5001,-372.9484 300,-362.9484 296.5001,-372.9485 303.5001,-372.9484"></polygon></a></g></g><!-- 94133550736256 --><g id="node13" class="node"><title>94133550736256</title><g id="a_node13"><a xlink:title="xla_computation(f)/sinop_type: sinsource: test_hlo.py:20"><polygon fill="#ffffff" stroke="#000000" points="404.1836,-290.6377 195.8164,-290.6377 195.8164,-254.6377 404.1836,-254.6377 404.1836,-290.6377"></polygon><text text-anchor="start" x="276.4772" y="-275.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">bitcast.7</text><text text-anchor="start" x="203.6583" y="-261.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[3,6,7,8,9]{4,1,3,2,0:T(8,128)}</text></a></g></g><!-- 94133550735360&#45;&gt;94133550736256 --><g id="edge12" class="edge"><title>94133550735360-&gt;94133550736256</title><g id="a_edge12"><a xlink:title="sine.3 -> bitcast.7"><path fill="none" stroke="#000000" d="M300,-326.4691C300,-318.7687 300,-309.612 300,-301.0543"></path><polygon fill="#000000" stroke="#000000" points="303.5001,-301.0509 300,-291.051 296.5001,-301.051 303.5001,-301.0509"></polygon></a></g></g><!-- 94133550740736 --><g id="node14" class="node"><title>94133550740736</title><polygon fill="#c8e6c9" stroke="#97b498" points="404.1836,-218.6377 195.8164,-218.6377 195.8164,-182.6377 404.1836,-182.6377 404.1836,-218.6377"></polygon><text text-anchor="start" x="281.1427" y="-203.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">copy.9</text><text text-anchor="start" x="203.6583" y="-189.0377" font-family="Times,serif" font-size="14.00" fill="#000000">f32[3,6,7,8,9]{4,3,2,1,0:T(8,128)}</text></g><!-- 94133550736256&#45;&gt;94133550740736 --><g id="edge13" class="edge"><title>94133550736256-&gt;94133550740736</title><g id="a_edge13"><a xlink:title="bitcast.7 -> copy.9"><path fill="none" stroke="#000000" d="M300,-254.4691C300,-246.7687 300,-237.612 300,-229.0543"></path><polygon fill="#000000" stroke="#000000" points="303.5001,-229.0509 300,-219.051 296.5001,-229.051 303.5001,-229.0509"></polygon></a></g></g><!-- 94133550741632 --><g id="node15" class="node"><title>94133550741632</title><polygon fill="#ffffff" stroke="#000000" points="351.3035,-146.6377 248.6965,-146.6377 248.6965,-110.6377 351.3035,-110.6377 351.3035,-146.6377"></polygon><text text-anchor="start" x="280.7521" y="-131.0377" font-family="Times,serif" font-weight="bold" font-size="14.00" fill="#000000">tuple.4</text><text text-anchor="start" x="256.8485" y="-117.0377" font-family="Times,serif" font-size="14.00" fill="#000000">(f32[3,6,7,8,9])</text></g><!-- 94133550740736&#45;&gt;94133550741632 --><g id="edge14" class="edge"><title>94133550740736-&gt;94133550741632</title><g id="a_edge14"><a xlink:title="copy.9 -> tuple.4"><path fill="none" stroke="#000000" d="M300,-182.4691C300,-174.7687 300,-165.612 300,-157.0543"></path><polygon fill="#000000" stroke="#000000" points="303.5001,-157.0509 300,-147.051 296.5001,-157.051 303.5001,-157.0509"></polygon></a></g></g><!-- cluster_94133549828960 --><g id="node16" class="node"><title>cluster_94133549828960</title><g id="a_node16"><a xlink:title=" "><ellipse fill="#bcaaa4" stroke="#8c7b75" cx="300" cy="-37.3188" rx="37.1386" ry="37.1386"></ellipse><text text-anchor="start" x="280.9488" y="-33.1188" font-family="Times,serif" font-size="14.00" fill="#000000">ROOT</text></a></g></g><!-- 94133550741632&#45;&gt;cluster_94133549828960 --><g id="edge15" class="edge"><title>94133550741632-&gt;cluster_94133549828960</title><g id="a_edge15"><a xlink:title=" "><path fill="none" stroke="#000000" d="M300,-110.596C300,-103.1939 300,-94.2407 300,-85.1718"></path><polygon fill="#000000" stroke="#000000" points="303.5001,-84.96 300,-74.9601 296.5001,-84.9601 303.5001,-84.96"></polygon></a></g></g></g>
<style>
@import url(https://fonts.googleapis.com/css?family=Roboto:400,700);
svg text {
  font-family: 'Roboto';
  font-size: 12px;
}

#node14:hover ~ #edge14 text { fill: #1976d2; }
#node14:hover ~ #edge14 path { stroke: #1976d2; stroke-width: .2em; }
#node14:hover ~ #edge14 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node15:hover ~ #edge14 text { fill: #d32f2f; }
#node15:hover ~ #edge14 path { stroke: #d32f2f; stroke-width: .2em; }
#node15:hover ~ #edge14 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node2:hover ~ #edge1 text { fill: #1976d2; }
#node2:hover ~ #edge1 path { stroke: #1976d2; stroke-width: .2em; }
#node2:hover ~ #edge1 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node3:hover ~ #edge1 text { fill: #d32f2f; }
#node3:hover ~ #edge1 path { stroke: #d32f2f; stroke-width: .2em; }
#node3:hover ~ #edge1 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node1:hover ~ #edge5 text { fill: #1976d2; }
#node1:hover ~ #edge5 path { stroke: #1976d2; stroke-width: .2em; }
#node1:hover ~ #edge5 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node7:hover ~ #edge5 text { fill: #d32f2f; }
#node7:hover ~ #edge5 path { stroke: #d32f2f; stroke-width: .2em; }
#node7:hover ~ #edge5 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#clust1:hover ~ #edge5 text { fill: #d32f2f; }
#clust1:hover ~ #edge5 path { stroke: #d32f2f; stroke-width: .2em; }
#clust1:hover ~ #edge5 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node8:hover ~ #edge7 text { fill: #1976d2; }
#node8:hover ~ #edge7 path { stroke: #1976d2; stroke-width: .2em; }
#node8:hover ~ #edge7 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node9:hover ~ #edge7 text { fill: #d32f2f; }
#node9:hover ~ #edge7 path { stroke: #d32f2f; stroke-width: .2em; }
#node9:hover ~ #edge7 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node5:hover ~ #edge4 text { fill: #1976d2; }
#node5:hover ~ #edge4 path { stroke: #1976d2; stroke-width: .2em; }
#node5:hover ~ #edge4 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node6:hover ~ #edge4 text { fill: #d32f2f; }
#node6:hover ~ #edge4 path { stroke: #d32f2f; stroke-width: .2em; }
#node6:hover ~ #edge4 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node15:hover ~ #edge15 text { fill: #1976d2; }
#node15:hover ~ #edge15 path { stroke: #1976d2; stroke-width: .2em; }
#node15:hover ~ #edge15 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node16:hover ~ #edge15 text { fill: #d32f2f; }
#node16:hover ~ #edge15 path { stroke: #d32f2f; stroke-width: .2em; }
#node16:hover ~ #edge15 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node7:hover ~ #edge6 text { fill: #1976d2; }
#node7:hover ~ #edge6 path { stroke: #1976d2; stroke-width: .2em; }
#node7:hover ~ #edge6 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node8:hover ~ #edge6 text { fill: #d32f2f; }
#node8:hover ~ #edge6 path { stroke: #d32f2f; stroke-width: .2em; }
#node8:hover ~ #edge6 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#clust3:hover ~ #edge6 text { fill: #d32f2f; }
#clust3:hover ~ #edge6 path { stroke: #d32f2f; stroke-width: .2em; }
#clust3:hover ~ #edge6 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node3:hover ~ #edge2 text { fill: #1976d2; }
#node3:hover ~ #edge2 path { stroke: #1976d2; stroke-width: .2em; }
#node3:hover ~ #edge2 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node4:hover ~ #edge2 text { fill: #d32f2f; }
#node4:hover ~ #edge2 path { stroke: #d32f2f; stroke-width: .2em; }
#node4:hover ~ #edge2 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#clust1:hover ~ #edge2 text { fill: #d32f2f; }
#clust1:hover ~ #edge2 path { stroke: #d32f2f; stroke-width: .2em; }
#clust1:hover ~ #edge2 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node9:hover ~ #edge8 text { fill: #1976d2; }
#node9:hover ~ #edge8 path { stroke: #1976d2; stroke-width: .2em; }
#node9:hover ~ #edge8 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node10:hover ~ #edge8 text { fill: #d32f2f; }
#node10:hover ~ #edge8 path { stroke: #d32f2f; stroke-width: .2em; }
#node10:hover ~ #edge8 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node11:hover ~ #edge11 text { fill: #1976d2; }
#node11:hover ~ #edge11 path { stroke: #1976d2; stroke-width: .2em; }
#node11:hover ~ #edge11 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node12:hover ~ #edge11 text { fill: #d32f2f; }
#node12:hover ~ #edge11 path { stroke: #d32f2f; stroke-width: .2em; }
#node12:hover ~ #edge11 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node6:hover ~ #edge9 text { fill: #1976d2; }
#node6:hover ~ #edge9 path { stroke: #1976d2; stroke-width: .2em; }
#node6:hover ~ #edge9 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node11:hover ~ #edge9 text { fill: #d32f2f; }
#node11:hover ~ #edge9 path { stroke: #d32f2f; stroke-width: .2em; }
#node11:hover ~ #edge9 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#clust2:hover ~ #edge9 text { fill: #1976d2; }
#clust2:hover ~ #edge9 path { stroke: #1976d2; stroke-width: .2em; }
#clust2:hover ~ #edge9 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node10:hover ~ #edge10 text { fill: #1976d2; }
#node10:hover ~ #edge10 path { stroke: #1976d2; stroke-width: .2em; }
#node10:hover ~ #edge10 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node11:hover ~ #edge10 text { fill: #d32f2f; }
#node11:hover ~ #edge10 path { stroke: #d32f2f; stroke-width: .2em; }
#node11:hover ~ #edge10 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#clust3:hover ~ #edge10 text { fill: #1976d2; }
#clust3:hover ~ #edge10 path { stroke: #1976d2; stroke-width: .2em; }
#clust3:hover ~ #edge10 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node12:hover ~ #edge12 text { fill: #1976d2; }
#node12:hover ~ #edge12 path { stroke: #1976d2; stroke-width: .2em; }
#node12:hover ~ #edge12 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node13:hover ~ #edge12 text { fill: #d32f2f; }
#node13:hover ~ #edge12 path { stroke: #d32f2f; stroke-width: .2em; }
#node13:hover ~ #edge12 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node4:hover ~ #edge3 text { fill: #1976d2; }
#node4:hover ~ #edge3 path { stroke: #1976d2; stroke-width: .2em; }
#node4:hover ~ #edge3 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node5:hover ~ #edge3 text { fill: #d32f2f; }
#node5:hover ~ #edge3 path { stroke: #d32f2f; stroke-width: .2em; }
#node5:hover ~ #edge3 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#clust2:hover ~ #edge3 text { fill: #d32f2f; }
#clust2:hover ~ #edge3 path { stroke: #d32f2f; stroke-width: .2em; }
#clust2:hover ~ #edge3 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#node13:hover ~ #edge13 text { fill: #1976d2; }
#node13:hover ~ #edge13 path { stroke: #1976d2; stroke-width: .2em; }
#node13:hover ~ #edge13 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }

#node14:hover ~ #edge13 text { fill: #d32f2f; }
#node14:hover ~ #edge13 path { stroke: #d32f2f; stroke-width: .2em; }
#node14:hover ~ #edge13 polygon { fill: #d32f2f; stroke: #d32f2f; stroke-width: .2em; }

#clust1:hover ~ #edge13 text { fill: #1976d2; }
#clust1:hover ~ #edge13 path { stroke: #1976d2; stroke-width: .2em; }
#clust1:hover ~ #edge13 polygon { fill: #1976d2; stroke: #1976d2; stroke-width: .2em; }
</style></svg></div>

<script>
var data = `
digraph G {
rankdir = TB;
compound = true;
label = <<b><br/>Computation xla_computation_f.14</b>>;
labelloc = t;
// Disable the tooltip.  Interestingly, "" doesn't work!
tooltip = " ";
// DOT graphs accept a stylesheet as a URI.  So naturally, an inline
// stylesheet is a data URI!
stylesheet=<
data:text/css,
@import url(https://fonts.googleapis.com/css?family=Roboto:400,700);
svg text {
  font-family: 'Roboto';
  font-size: 12px;
}

%23node14:hover ~ %23edge14 text { fill: %231976d2; }
%23node14:hover ~ %23edge14 path { stroke: %231976d2; stroke-width: .2em; }
%23node14:hover ~ %23edge14 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node15:hover ~ %23edge14 text { fill: %23d32f2f; }
%23node15:hover ~ %23edge14 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node15:hover ~ %23edge14 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node2:hover ~ %23edge1 text { fill: %231976d2; }
%23node2:hover ~ %23edge1 path { stroke: %231976d2; stroke-width: .2em; }
%23node2:hover ~ %23edge1 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node3:hover ~ %23edge1 text { fill: %23d32f2f; }
%23node3:hover ~ %23edge1 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node3:hover ~ %23edge1 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node1:hover ~ %23edge5 text { fill: %231976d2; }
%23node1:hover ~ %23edge5 path { stroke: %231976d2; stroke-width: .2em; }
%23node1:hover ~ %23edge5 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node7:hover ~ %23edge5 text { fill: %23d32f2f; }
%23node7:hover ~ %23edge5 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node7:hover ~ %23edge5 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23clust1:hover ~ %23edge5 text { fill: %23d32f2f; }
%23clust1:hover ~ %23edge5 path { stroke: %23d32f2f; stroke-width: .2em; }
%23clust1:hover ~ %23edge5 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node8:hover ~ %23edge7 text { fill: %231976d2; }
%23node8:hover ~ %23edge7 path { stroke: %231976d2; stroke-width: .2em; }
%23node8:hover ~ %23edge7 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node9:hover ~ %23edge7 text { fill: %23d32f2f; }
%23node9:hover ~ %23edge7 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node9:hover ~ %23edge7 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node5:hover ~ %23edge4 text { fill: %231976d2; }
%23node5:hover ~ %23edge4 path { stroke: %231976d2; stroke-width: .2em; }
%23node5:hover ~ %23edge4 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node6:hover ~ %23edge4 text { fill: %23d32f2f; }
%23node6:hover ~ %23edge4 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node6:hover ~ %23edge4 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node15:hover ~ %23edge15 text { fill: %231976d2; }
%23node15:hover ~ %23edge15 path { stroke: %231976d2; stroke-width: .2em; }
%23node15:hover ~ %23edge15 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node16:hover ~ %23edge15 text { fill: %23d32f2f; }
%23node16:hover ~ %23edge15 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node16:hover ~ %23edge15 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node7:hover ~ %23edge6 text { fill: %231976d2; }
%23node7:hover ~ %23edge6 path { stroke: %231976d2; stroke-width: .2em; }
%23node7:hover ~ %23edge6 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node8:hover ~ %23edge6 text { fill: %23d32f2f; }
%23node8:hover ~ %23edge6 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node8:hover ~ %23edge6 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23clust3:hover ~ %23edge6 text { fill: %23d32f2f; }
%23clust3:hover ~ %23edge6 path { stroke: %23d32f2f; stroke-width: .2em; }
%23clust3:hover ~ %23edge6 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node3:hover ~ %23edge2 text { fill: %231976d2; }
%23node3:hover ~ %23edge2 path { stroke: %231976d2; stroke-width: .2em; }
%23node3:hover ~ %23edge2 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node4:hover ~ %23edge2 text { fill: %23d32f2f; }
%23node4:hover ~ %23edge2 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node4:hover ~ %23edge2 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23clust1:hover ~ %23edge2 text { fill: %23d32f2f; }
%23clust1:hover ~ %23edge2 path { stroke: %23d32f2f; stroke-width: .2em; }
%23clust1:hover ~ %23edge2 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node9:hover ~ %23edge8 text { fill: %231976d2; }
%23node9:hover ~ %23edge8 path { stroke: %231976d2; stroke-width: .2em; }
%23node9:hover ~ %23edge8 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node10:hover ~ %23edge8 text { fill: %23d32f2f; }
%23node10:hover ~ %23edge8 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node10:hover ~ %23edge8 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node11:hover ~ %23edge11 text { fill: %231976d2; }
%23node11:hover ~ %23edge11 path { stroke: %231976d2; stroke-width: .2em; }
%23node11:hover ~ %23edge11 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node12:hover ~ %23edge11 text { fill: %23d32f2f; }
%23node12:hover ~ %23edge11 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node12:hover ~ %23edge11 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node6:hover ~ %23edge9 text { fill: %231976d2; }
%23node6:hover ~ %23edge9 path { stroke: %231976d2; stroke-width: .2em; }
%23node6:hover ~ %23edge9 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node11:hover ~ %23edge9 text { fill: %23d32f2f; }
%23node11:hover ~ %23edge9 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node11:hover ~ %23edge9 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23clust2:hover ~ %23edge9 text { fill: %231976d2; }
%23clust2:hover ~ %23edge9 path { stroke: %231976d2; stroke-width: .2em; }
%23clust2:hover ~ %23edge9 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node10:hover ~ %23edge10 text { fill: %231976d2; }
%23node10:hover ~ %23edge10 path { stroke: %231976d2; stroke-width: .2em; }
%23node10:hover ~ %23edge10 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node11:hover ~ %23edge10 text { fill: %23d32f2f; }
%23node11:hover ~ %23edge10 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node11:hover ~ %23edge10 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23clust3:hover ~ %23edge10 text { fill: %231976d2; }
%23clust3:hover ~ %23edge10 path { stroke: %231976d2; stroke-width: .2em; }
%23clust3:hover ~ %23edge10 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node12:hover ~ %23edge12 text { fill: %231976d2; }
%23node12:hover ~ %23edge12 path { stroke: %231976d2; stroke-width: .2em; }
%23node12:hover ~ %23edge12 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node13:hover ~ %23edge12 text { fill: %23d32f2f; }
%23node13:hover ~ %23edge12 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node13:hover ~ %23edge12 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node4:hover ~ %23edge3 text { fill: %231976d2; }
%23node4:hover ~ %23edge3 path { stroke: %231976d2; stroke-width: .2em; }
%23node4:hover ~ %23edge3 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node5:hover ~ %23edge3 text { fill: %23d32f2f; }
%23node5:hover ~ %23edge3 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node5:hover ~ %23edge3 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23clust2:hover ~ %23edge3 text { fill: %23d32f2f; }
%23clust2:hover ~ %23edge3 path { stroke: %23d32f2f; stroke-width: .2em; }
%23clust2:hover ~ %23edge3 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23node13:hover ~ %23edge13 text { fill: %231976d2; }
%23node13:hover ~ %23edge13 path { stroke: %231976d2; stroke-width: .2em; }
%23node13:hover ~ %23edge13 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }

%23node14:hover ~ %23edge13 text { fill: %23d32f2f; }
%23node14:hover ~ %23edge13 path { stroke: %23d32f2f; stroke-width: .2em; }
%23node14:hover ~ %23edge13 polygon { fill: %23d32f2f; stroke: %23d32f2f; stroke-width: .2em; }

%23clust1:hover ~ %23edge13 text { fill: %231976d2; }
%23clust1:hover ~ %23edge13 path { stroke: %231976d2; stroke-width: .2em; }
%23clust1:hover ~ %23edge13 polygon { fill: %231976d2; stroke: %231976d2; stroke-width: .2em; }
>

94133550737152 [label=<<b>Parameter 1</b><br/>f32[7,8,4,9,5]{3,2,4,1,0:T(4,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#c88719", fillcolor="#ffb74d"];
94133550738048 [label=<<b>Parameter 0</b><br/>f32[3,4,5,6]{3,1,2,0:T(4,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#cbae82", fillcolor="#ffe0b2"];
94133550738944 [label=<<b>bitcast</b><br/>f32[3,4,5,6,1,1]{3,1,5,4,2,0:T(4,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="black", fillcolor="white"];
subgraph cluster_94133549829168 {
style="rounded,filled,bold"; fillcolor="#f5f5f5"; color="#c2c2c2;"
label = <Fused expression for <b>fusion.6</b><br/>output fusion<br/>kind=kOutput>;
labelloc = t;
tooltip = " ";
94133550109056 [label=<<b>Parameter 1</b><br/>f32[3,4,5,6,1,1]{3,1,5,4,2,0:T(4,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#cbae82", fillcolor="#ffe0b2"];
subgraph cluster_94133549829376 {
style="rounded,filled,bold"; fillcolor="#f5f5f5"; color="#c2c2c2;"
label = <Fused expression for <b>fusion.9</b><br/>loop fusion<br/>kind=kLoop>;
labelloc = t;
tooltip = " ";
94133550107264 [label=<<b>Parameter 0</b><br/>f32[3,4,5,6,1,1]{3,1,5,4,2,0:T(4,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#cbae82", fillcolor="#ffe0b2"];
94133550108160 [label=<<b>copy.19</b><br/>bf16[3,4,5,6,1,1]{3,2,1,5,4,0:T(8,128)(2,1)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#97b498", fillcolor="#c8e6c9"];

}  // cluster_94133549829376

94133550110848 [label=<<b>Parameter 0</b><br/>f32[7,8,4,9,5]{3,2,4,1,0:T(4,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#c88719", fillcolor="#ffb74d"];
subgraph cluster_94133549829584 {
style="rounded,filled,bold"; fillcolor="#f5f5f5"; color="#c2c2c2;"
label = <Fused expression for <b>fusion.8</b><br/>loop fusion<br/>kind=kLoop>;
labelloc = t;
tooltip = " ";
94133550104576 [label=<<b>Parameter 0</b><br/>f32[7,8,4,9,5]{3,2,4,1,0:T(4,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#c88719", fillcolor="#ffb74d"];
94133550105472 [label=<<b>copy.17</b><br/>bf16[7,8,4,9,5]{3,4,2,1,0:T(8,128)(2,1)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#97b498", fillcolor="#c8e6c9"];
94133550106368 [label=<<b>bitcast.9</b><br/>bf16[7,8,4,9,5,1]{3,4,2,1,0,5:T(8,128)(2,1)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="black", fillcolor="white"];

}  // cluster_94133549829584

94133550498816 [label=<<b>convolution.5</b><br/>window={size=1x7x8x4 pad=0_0x6_6x7_7x0_0 rhs_reversal=0x1x1x0}<br/>dim_labels=03fb12_123oi0-&gt;0b12f3<br/>f32[3,6,7,8,9,1]{4,1,5,3,2,0:T(8,128)}>, shape=rect, tooltip="", style="filled", fontcolor="white", color="#003c8f", fillcolor="#1565c0"];
94133550735360 [label=<<b>sine.3</b><br/>f32[3,6,7,8,9,1]{4,1,5,3,2,0:T(8,128)}>, shape=rect, tooltip="xla_computation(f)/sin
op_type: sin
source: test_hlo.py:20", style="filled", fontcolor="black", color="#cbc693", fillcolor="#fff9c4"];
94133550736256 [label=<<b>bitcast.7</b><br/>f32[3,6,7,8,9]{4,1,3,2,0:T(8,128)}>, shape=rect, tooltip="xla_computation(f)/sin
op_type: sin
source: test_hlo.py:20", style="filled", fontcolor="black", color="black", fillcolor="white"];

}  // cluster_94133549829168

94133550740736 [label=<<b>copy.9</b><br/>f32[3,6,7,8,9]{4,3,2,1,0:T(8,128)}>, shape=rect, tooltip="", style="filled", fontcolor="black", color="#97b498", fillcolor="#c8e6c9"];
94133550741632 [label=<<b>tuple.4</b><br/>(f32[3,6,7,8,9])>, shape=rect, tooltip="", style="filled", fontcolor="black", color="black", fillcolor="white"];
cluster_94133549828960 [label=<ROOT>, shape=circle, tooltip=" ", style="filled", fontcolor="black", color="#8c7b75", fillcolor="#bcaaa4"];
94133550738048 -> 94133550738944 [arrowhead=empty tooltip="parameter.1 -> bitcast" ];
94133550738944 -> 94133550109056 [arrowhead=empty tooltip="bitcast -> param_1.9" ];
94133550109056 -> 94133550107264 [arrowhead=empty tooltip="param_1.9 -> param_0.17" ];
94133550107264 -> 94133550108160 [arrowhead=empty tooltip="param_0.17 -> copy.19" ];
94133550737152 -> 94133550110848 [arrowhead=normal tooltip="parameter.2 -> param_0.16" ];
94133550110848 -> 94133550104576 [arrowhead=normal tooltip="param_0.16 -> param_0.15" ];
94133550104576 -> 94133550105472 [arrowhead=normal tooltip="param_0.15 -> copy.17" ];
94133550105472 -> 94133550106368 [arrowhead=normal tooltip="copy.17 -> bitcast.9" ];
94133550108160 -> 94133550498816 [arrowhead=empty tooltip="copy.19 -> convolution.5"  headlabel="0", labeldistance=2];
94133550106368 -> 94133550498816 [arrowhead=normal tooltip="bitcast.9 -> convolution.5"  headlabel="1", labeldistance=2];
94133550498816 -> 94133550735360 [arrowhead=normal tooltip="convolution.5 -> sine.3" ];
94133550735360 -> 94133550736256 [arrowhead=normal tooltip="sine.3 -> bitcast.7" ];
94133550736256 -> 94133550740736 [arrowhead=normal tooltip="bitcast.7 -> copy.9" ];
94133550740736 -> 94133550741632 [arrowhead=normal tooltip="copy.9 -> tuple.4" ];
94133550741632 -> cluster_94133549828960 [tooltip=" "];
}`;
var cssregex = new RegExp('stylesheet=<([^]*)\n>\n', 'gm');
var results = cssregex.exec(data)
// graphviz has problem dealing with large stylesheets.
// https://github.com/tensorflow/tensorflow/issues/17220#issuecomment-369228492
// In order to avoid the problem, remove the stylesheet from the dot and
// insert it directly info the rendered SVG.
var dot_data = data;
var css_data = ''
if (results !== null) {
  css_data = results[1].replace(/\s*data:.*\s*,/,''); // Strip content-type field.
  // CSS inside DOT is URL-escaped, so we must unescape it
  // before we can insert it into SVG.
  css_data = unescape(css_data);
  dot_data = data.replace(cssregex, ''); // Remove the stylesheet
}
</script>

(Written on 1 March 2022)
