## Introduction
The core function of data analysis for plot or data process used by SZQ lab from China Agricultural University


## Example usage


```python
from Bio import Phylo
import matplotlib as mpl
import matplotlib.pyplot as plt
from io import StringIO
import matplotlib.collections as mpcollections
from copy import copy

import pandas as pd
import numpy as np
import seaborn as sn

from cvmcore.cvmcore import cvmplot

from scipy.cluster.hierarchy import linkage, dendrogram, complete, to_tree
from scipy.spatial.distance import squareform
```


```python
mlst = [[np.nan, 19., 12.,  9.,  5.,  9.,  2.],
        [np.nan, 19., 12.,  9.,  5.,  9.,  2.],
        [10., 17., 12.,  9., np.nan,  9.,  2.],
        [10., 19., 12., np.nan,  5.,  9.,  2.],
        [np.nan, 19., 13.,  9.,  5.,  9.,  2.]]
genes = np.char.replace(np.array(np.arange(1, 8), dtype='str'), '', 'gene_', count=1)
samples = np.char.replace(np.array(np.arange(1, 6), dtype='str'), '', 'sample_', count=1)
df_mlst = pd.DataFrame(mlst, index=samples, columns=genes)
diff_matrix = cvmplot.get_diff_df(df_mlst)
```


```python
diff_matrix
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sample_1</th>
      <th>sample_2</th>
      <th>sample_3</th>
      <th>sample_4</th>
      <th>sample_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sample_1</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_3</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>sample_4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>sample_5</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
link_matrix =linkage(squareform(diff_matrix), method='complete')
```


```python
link_matrix
```




    array([[0., 1., 0., 2.],
           [3., 5., 0., 3.],
           [2., 6., 1., 4.],
           [4., 7., 2., 5.]])



### Plot a rectangular dendrogram


```python
fig, ax= plt.subplots(1,1)
lableorder, ax = cvmplot.rectree(link_matrix, scale_max=7, labels=samples, ax=ax)
fig.tight_layout()
fig.savefig('Screenshots/dendrogram.png')
```


![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_7_0.png)


### Plot rectangular dendrogram with heatmap


```python
#create dataframe
mat = np.random.randint(70, 100, (5, 10))
loci = np.char.replace(np.array(np.arange(1, 11), dtype='str'), '', 'loci_', count=1)
sample = np.char.replace(np.array(np.arange(1, 6), dtype='str'), '', 'sample', count=1)
df_heatmap = pd.DataFrame(mat, index=sample, columns=loci)
```


```python
#create linkage matrix
diff_matrix = [[0, 0, 1, 0, 1],
               [0, 0, 1, 0, 1],
               [1, 1, 0, 1, 2],
               [0, 0, 1, 0, 1],
               [1, 1, 2, 1, 0]]

linkage_matrix = linkage(squareform(diff_matrix),'complete')
```


```python
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,3), gridspec_kw={'width_ratios': [1, 2]})
fig.tight_layout(w_pad=-2)

row_order, ax1 = cvmplot.rectree(linkage_matrix,labels=sample, no_labels=True, scale_max=3, ax=ax1)
cvmplot.heatmap(df_heatmap, order=row_order, ax=ax2, cbar=True)

ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=15)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=15)
ax2.xaxis.tick_top()

fig.savefig('Screenshots/test.pdf')
```

    [ 5 15 25 35 45]
    ['sample5', 'sample3', 'sample4', 'sample1', 'sample2']



![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_11_1.png)


#### set minimum value of heatmap


```python
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,2.4), gridspec_kw={'width_ratios': [1, 2]})
fig.tight_layout(pad=-2)

order, ax1 = cvmplot.rectree(linkage_matrix,labels=sample, no_labels=True, scale_max=3, ax=ax1)
cvmplot.heatmap(df_heatmap, order=order, ax=ax2, cbar=True, vmin=90)

ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=15)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=15)
ax2.xaxis.tick_top()

fig.savefig('Screenshots/dendrogram_heatmap_minimumvalue.pdf')
```

    [ 5 15 25 35 45]
    ['sample5', 'sample3', 'sample4', 'sample1', 'sample2']



![png](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_13_1.png)


#### using cmap to change color


```python
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(8,2.4), gridspec_kw={'width_ratios': [1, 2]})
fig.tight_layout(pad=-2)

order, ax1 = cvmplot.rectree(linkage_matrix,labels=sample, no_labels=True, scale_max=3, ax=ax1)
cvmplot.heatmap(df_heatmap, order=order, ax=ax2, cmap='tab20', cbar=True)

ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=15)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=15)
ax2.xaxis.tick_top()
fig.savefig('Screenshots/dendrogram_heatmap_cmap.pdf')
```

    [ 5 15 25 35 45]
    ['sample5', 'sample3', 'sample4', 'sample1', 'sample2']



![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_15_1.png)


### Plot a circular dendrogram


```python
# generate two clusters: a with 100 points, b with 50:
np.random.seed(4711)  # for repeatability of this tutorial
a = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100,])
b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[50,])
X = np.concatenate((a, b),)
```


```python
Z = linkage(X, 'ward')
```


```python
Z2 = dendrogram(Z, no_plot=True)
```


```python
# set open angle
fig, ax= plt.subplots(1,1,figsize=(10,10))

cvmplot.circulartree(Z2,addlabels=True, fontsize=10, ax=ax)
fig.tight_layout()
fig.savefig('Screenshots/circular_dendrogram.png')
```


![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_20_0.png)


#### color label


```python
colors = [{'#0070c7':'2021'}, {'#3a9245':'2022'}, {'#f8d438':'2023'}]
result = np.random.choice(colors, size=150)
label_colors_map = dict(zip(Z2['ivl'], result))
point_colors_map = dict(zip(Z2['ivl'], result))
```


```python
fig, ax= plt.subplots(1,1,figsize=(10,10))
cvmplot.circulartree(Z2, addlabels=True, branch_color=False, label_colors= label_colors_map, fontsize=15)
fig.tight_layout()
fig.savefig('Screenshots/circular_dendrogram_color_label.png')
```


![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_23_0.png)


#### set open angle


```python
fig, ax= plt.subplots(1,1,figsize=(10,10))
cvmplot.circulartree(Z2, addlabels=True, branch_color=False, label_colors= label_colors_map, fontsize=15, open_angle=30)
fig.tight_layout()
fig.savefig('Screenshots/circular_dendrogram_openangle.png')
```


![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_25_0.png)


#### set start angle


```python
fig, ax= plt.subplots(1,1,figsize=(10,10))
cvmplot.circulartree(Z2, addlabels=True, branch_color=False, label_colors= label_colors_map, fontsize=15, open_angle=90,
                     start_angle=30
                    )
fig.tight_layout()
fig.savefig('Screenshots/circular_dendrogram_startangle.png')
```


![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_27_0.png)


#### add point 


```python
fig, ax= plt.subplots(1,1,figsize=(12,10))
cvmplot.circulartree(Z2, addlabels=True, branch_color=False, label_colors= label_colors_map, fontsize=15, addpoints=True,
                     point_colors = point_colors_map, point_legend_title='Species', pointsize=25)
fig.tight_layout()
fig.savefig('Screenshots/circular_dendrogram_tippoints.png')
```


![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_29_0.png)



### Plot phylogenetic tree


```python
tree = "(((A:0.2, B:0.3):0.3,(C:0.5, D:0.3):0.2):0.3, E:0.7):1.0;"
tree = Phylo.read(StringIO(tree), 'newick')
```


```python
fig, ax= plt.subplots(1,1, figsize=(10, 10))
ax, lable_order = cvmplot.phylotree(tree=tree, color='k', lw=1, ax=ax, show_label=True, align_label=True, labelsize=15)
fig.tight_layout()
fig.savefig('Screenshots/phylogenetic tree.png')
```


![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_35_0.png)


#### Plot tree with heatmap


```python
#create dataframe
mat = np.random.randint(70, 100, (5, 10))
col = np.char.replace(np.array(np.arange(1, 11), dtype='str'), '', 'column_', count=1)
strains = ['A', 'B', 'C', 'D', 'E']
df_heatmap = pd.DataFrame(mat, index=strains, columns=col)
```


```python
df_heatmap
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>column_1</th>
      <th>column_2</th>
      <th>column_3</th>
      <th>column_4</th>
      <th>column_5</th>
      <th>column_6</th>
      <th>column_7</th>
      <th>column_8</th>
      <th>column_9</th>
      <th>column_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>89</td>
      <td>73</td>
      <td>91</td>
      <td>75</td>
      <td>95</td>
      <td>90</td>
      <td>93</td>
      <td>74</td>
      <td>99</td>
      <td>97</td>
    </tr>
    <tr>
      <th>B</th>
      <td>73</td>
      <td>90</td>
      <td>75</td>
      <td>89</td>
      <td>85</td>
      <td>72</td>
      <td>82</td>
      <td>85</td>
      <td>96</td>
      <td>82</td>
    </tr>
    <tr>
      <th>C</th>
      <td>84</td>
      <td>82</td>
      <td>86</td>
      <td>74</td>
      <td>72</td>
      <td>75</td>
      <td>91</td>
      <td>83</td>
      <td>97</td>
      <td>98</td>
    </tr>
    <tr>
      <th>D</th>
      <td>72</td>
      <td>77</td>
      <td>72</td>
      <td>98</td>
      <td>79</td>
      <td>73</td>
      <td>87</td>
      <td>91</td>
      <td>98</td>
      <td>94</td>
    </tr>
    <tr>
      <th>E</th>
      <td>88</td>
      <td>75</td>
      <td>88</td>
      <td>73</td>
      <td>77</td>
      <td>72</td>
      <td>74</td>
      <td>73</td>
      <td>99</td>
      <td>86</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig,(ax1, ax2)= plt.subplots(1,2, figsize=(10, 3), gridspec_kw={'width_ratios':[1, 2]})
fig.tight_layout(pad=-2)
ax1, order = cvmplot.phylotree(tree=tree, color='k', lw=1, ax=ax1, show_label=True, align_label=True, labelsize=15)
cvmplot.heatmap(df_heatmap, order=order, ax=ax2, cbar=True, vmin=90)

ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=15)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=15)
ax2.xaxis.tick_top()

fig.savefig('Screenshots/phylotree_with_heatmap.pdf')
```




![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_39_1.png)


#### remove labels at the tip of the tree


```python
fig,(ax1, ax2)= plt.subplots(1,2, figsize=(10, 3), gridspec_kw={'width_ratios':[1, 2]})
fig.tight_layout(pad=-2)
ax1, order = cvmplot.phylotree(tree=tree, color='k', lw=1, ax=ax1, show_label=False)
cvmplot.heatmap(df_heatmap, order=order, ax=ax2, cbar=True, vmin=90)

ax1.set_xticklabels(ax1.get_xticklabels(), fontsize=15)

ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, fontsize=15)
ax2.set_yticklabels(ax2.get_yticklabels(), fontsize=15)
ax2.xaxis.tick_top()

fig.savefig('Screenshots/phylotree_with_heatmap-remove_tiplable.pdf')
```



![image](http://microbe.genesclouds.com.cn/microbe/library/screenshots/output_41_1.png)

