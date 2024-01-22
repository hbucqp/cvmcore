import os
import sys

# data process
import dask.array as da
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, complete, to_tree
from scipy.spatial.distance import squareform
from tabulate import tabulate

# from Bio.Blast import NCBIWWW
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Blast import NCBIXML
from Bio.Blast.Applications import NcbiblastnCommandline
from Bio.Blast.Applications import NcbimakeblastdbCommandline


# matplotlib
import matplotlib
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import Affine2D
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import Optional, List, Dict, Union, Tuple

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['svg.fonttype'] = 'none'


class cvmbox():
    pass

    @staticmethod
    def single_cgmlst2ref(files_dir: str, outpath: str):
        """
        Create cgMLST reference sequences using fasta files downloaded from "https://www.cgmlst.org/"
        """
        files_dir = os.path.abspath(files_dir)
        new_records = []
        for file in os.listdir(files_dir):
            # print(file)
            if file.endswith('.fasta'):
                file_base = file.split('.')[0]
                # print(file_base)
                file = os.path.join(files_dir, file)
                records = SeqIO.parse(file, 'fasta')
                for record in records:
                    record.id = file_base + "_" + record.id
                    record.name = file_base + "_" + record.name
                    record.description = ''
                    # print(record.id)
                    # print(record)
                    new_records.append(record)

        # check if outpath exists
        outdir = os.path.abspath(outpath)
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)

        # Specify the output file name
        output_file = "reference.fa"
        output_file = os.path.join(outdir, output_file)
        # Write the modified sequences to the new fasta file
        with open(output_file, "w") as output_handle:
            SeqIO.write(new_records, output_handle, "fasta")


class cvmplot():
    pass

    @staticmethod
    def _auto_ticks(ax, labels, axis, shape):
        """Determine ticks and ticklabels that minimize overlap."""
        transform = ax.figure.dpi_scale_trans.inverted()
        bbox = ax.get_window_extent().transformed(transform)
        size = [bbox.width, bbox.height][axis]

        axis_tag = axis
        shape.reverse()
        start = shape[axis_tag]
        if axis_tag == 1:
            start = 0
        axis = [ax.xaxis, ax.yaxis][axis_tag]
        tick, = axis.set_ticks([0])
        fontsize = tick.label1.get_size()
        max_ticks = int(size // (fontsize / 72))
        if max_ticks < 1:
            return [], []
        tick_every = len(labels) // max_ticks + 1
        tick_every = 1 if tick_every == 0 else tick_every
        ticks, labels = cvmplot._skip_ticks(labels, tick_every, start, axis_tag)
        return ticks, labels

    def rectree(matrix,
                figsize: Optional[Tuple]=None,
                labels: Optional[List]=None,
                no_labels: bool=False,
                scale_max: float=10,
                ax=None):
        """
        Drawing a rectangular dendrogram using scipy dendrogram function.
        Parameters
        -----------
        matrix: linkage matrix
            A matrix returned by scipy.cluster.hierarchy.linkage.
        figsize: (x, y) tuple-like
            1D tuple-like of floats to specify the figure size.
        labels: list
            The list of the sample's name.
        scale_max: float
            The maximum value of the scale.
        ax : matplotlib Axes, optional
            Axes in which to draw the plot, otherwise use the currently-active Axes.


        Returns
        -------
        Raises
        ------
        Notes
        -----
        References
        ----------
        See Also
        --------
        Examples
        --------
        """
        if figsize == None:
            figsize = (15, 15)

        if ax is None:
            ax = plt.gca()
        else:
            ax=ax

        # MatrixS = matrix.shape

        # fig, ax = plt.subplots(1, 1, figsize=figsize)
        dendro_info = dendrogram(
            matrix, ax=ax, orientation='left', no_plot=False, labels=labels, no_labels=no_labels)
        order = dendro_info['ivl']

        # set intervals on axis
        ax.set_xticks(np.arange(0, scale_max, 1))
        ax.set_xlim(scale_max, 0)

        # move spines of ax
        ax.spines[['bottom', 'right', 'left']].set_visible(False)

        # move scale bar on top
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        # save plot
        # plt.tight_layout()
        # plt.show()
        return order, ax

    def circulartree(Z2,
                     fontsize: float=8,
                     open_angle: float=0,
                     start_angle: float=0,
                     figsize: Optional[Tuple]=None,
                     addpoints: bool=False,
                     pointsize: float=15,
                     point_colors: Optional[dict]=None,
                     point_legend_title: str='Category',
                     palette: str="gist_rainbow",
                     addlabels: bool=True,
                     label_colors: Optional[dict]=None,
                     show: bool=True,
                     branch_color: bool=False,
                     sample_classes: Optional[dict]=None,

                     colorlabels: Optional[dict]=None,
                     colorlabels_legend: Optional[dict]=None) -> plt.Axes:
        """
        Drawing a radial dendrogram from a scipy dendrogram output.
        Parameters
        ----------
        Z2 : dictionary
            A dictionary returned by scipy.cluster.hierarchy.dendrogram
        fontsize : float
            A float to specify the font size
        figsize : (x, y) tuple-like
            1D tuple-like of floats to specify the figure size
        palette : string
            Matplotlib colormap name.
        branch_color: bool
            whether or not render branch with colors.
        add_points: bool
            whether or not render leaf point with colors.
        point_colors: dict
            A dictionary to set the color of the leaf point. The Key is the name of the leaflabel.
            The value is the hex color.
            e.g., {'label1':{'#ffffff':'Category1'}, 'label2':{'##f77124':'Catogory2'}...}
        point_legend_title: str
            The title of leaf point legend.
        pointsize: float
            A float to specify the leaf point size
        sample_classes : dict
            A dictionary that contains lists of sample subtypes or classes. These classes appear
            as color labels of each leaf. Colormaps are automatically assigned. Not compatible
            with options "colorlabels" and "colorlabels_legend".
            e.g., {"color1":["Class1","Class2","Class1","Class3", ....]}
        start_angle : float
            The angle of the start point of the circular plot.
            e.g., range from 0 to 360.
        open_angle : float
            The angle of the endpoint of the circular plot.
            e.g., range from 0 to 360.
        addlabels: bool
            A bool to choose if labels are shown.
        label_colors: dict
            A dictionary to set the color of the leaf label. The Key is the name of the leaflabel.
            The value is the hex color.
            e.g., {'label1':'#ffffff', 'label2':'##f77124'...}
        colorlabels_legend : dict
            A nested dictionary to generate the legends of color labels. The key is the name of
            the color label. The value is a dictionary that has two keys "colors" and "labels".
            The value of "colors" is the list of RGB color codes, each corresponds to the class of a leaf.
            e.g., {"color1":{"colors":[[1,0,0,1], ....], "labels":["label1","label2",...]}}
        show : bool
            Whether or not to show the figure.
        Returns
        -------
        Raises
        ------
        Notes
        -----
        References
        ----------
        See Also
        --------
        Examples
        --------
        """
        if figsize == None and colorlabels != None:
            figsize = [7, 5]
        elif figsize == None and sample_classes != None:
            figsize = [7, 5]
        elif figsize == None:
            figsize = [10, 10]
        linewidth = 0.5
        R = 1
        width = R * 0.1
        space = R * 0.05
        if colorlabels != None:
            offset = width * len(colorlabels) / R + space * \
                (len(colorlabels) - 1) / R + 0.05
            print(offset)
        elif sample_classes != None:
            offset = width * len(sample_classes) / R + \
                space * (len(sample_classes) - 1) / R + 0.05
            print(offset)
        else:
            offset = 0

        xmax = np.amax(Z2['icoord'])
        ymax = np.amax(Z2['dcoord'])

        ucolors = sorted(set(Z2["color_list"]))
        # print(f'ucolors is {ucolors}')
        #cmap = cm.gist_rainbow(np.linspace(0, 1, len(ucolors)))
        cmp = plt.get_cmap(palette, len(ucolors))
        # print(cmp)
        if type(cmp) == LinearSegmentedColormap:
            cmap = cmp(np.linspace(0, 1, len(ucolors)))
        else:
            cmap = cmp.colors
        fig, ax = plt.subplots(figsize=figsize)
        i = 0
        label_coords = []
        leaf_coords = []
        check_coords = []

        # Get the xtick position and create iv_ticks array
        iv_ticks = np.arange(5, len(Z2['ivl']) * 10 + 5, 10)

        for x, y, c in sorted(zip(Z2['icoord'], Z2['dcoord'], Z2["color_list"])):
            if not branch_color:
                _color = 'black'
            else:
                _color = cmap[ucolors.index(c)]
            # _color = 'black'

            # np.abs(_xr1)<0.000000001 and np.abs(_yr1) <0.000000001:
            if c == "C0":
                # print('test')
                _color = "black"

            # transforming original x coordinates into relative circumference positions and y into radius
            # the rightmost leaf is going to [1, 0]
            r = R * (1 - np.array(y) / ymax)
            # _x=np.cos(2*np.pi*np.array([x[0],x[2]])/xmax) # transforming original x coordinates into x circumference positions
            _x = np.cos((2 * np.pi * (360 - open_angle) / 360)
                        * np.array([x[0], x[2]]) / xmax)
            _xr0 = _x[0] * r[0]
            _xr1 = _x[0] * r[1]
            _xr2 = _x[1] * r[2]
            _xr3 = _x[1] * r[3]
            # _y=np.sin(2*np.pi*np.array([x[0],x[2]])/xmax) # transforming original x coordinates into y circumference positions
            # transforming original x coordinates into y circumference positions
            _y = np.sin(2 * np.pi * (360 - open_angle) /
                        360 * np.array([x[0], x[2]]) / xmax)
            _yr0 = _y[0] * r[0]
            _yr1 = _y[0] * r[1]
            _yr2 = _y[1] * r[2]
            _yr3 = _y[1] * r[3]

            # calculate the new coordinate
            new_xr0, new_yr0 = cvmplot.rotate_point(_xr0, _yr0, start_angle)
            new_xr1, new_yr1 = cvmplot.rotate_point(_xr1, _yr1, start_angle)
            new_xr2, new_yr2 = cvmplot.rotate_point(_xr2, _yr2, start_angle)
            new_xr3, new_yr3 = cvmplot.rotate_point(_xr3, _yr3, start_angle)

            # plotting radial lines
            ax.plot([new_xr0, new_xr1], [new_yr0, new_yr1],
                    c=_color, linewidth=linewidth, rasterized=True)
            ax.plot([new_xr2, new_xr3], [new_yr2, new_yr3],
                    c=_color, linewidth=linewidth, rasterized=True)

            # plotting circular links between nodes
            if new_yr1 >= 0 and new_yr2 >= 0:
                link = np.sqrt(
                    r[1]**2 - np.linspace(new_xr1, new_xr2, 10000)**2)
                ax.plot(np.linspace(new_xr1, new_xr2, 10000), link,
                        c=_color, linewidth=linewidth, rasterized=True)
                # ax.plot(link, np.linspace(new_xr1, new_xr2, 10000),
                #         c=_color, linewidth=linewidth, rasterized=True)

            elif new_yr1 <= 0 and new_yr2 <= 0:
                link = -np.sqrt(r[1]**2 -
                                np.linspace(new_xr1, new_xr2, 10000)**2)

                ax.plot(np.linspace(new_xr1, new_xr2, 10000), link,
                        c=_color, linewidth=linewidth, rasterized=True)
            elif new_yr1 >= 0 and new_yr2 <= 0:
                _r = r[1]
                if new_xr1 < 0 or new_xr2 < 0:
                    _r = -_r
                link = np.sqrt(r[1]**2 -
                               np.linspace(new_xr1, _r, 10000)**2)
                # print(link)
                # print(dict(zip(np.linspace(_xr1, _r, 10000), link)))
                ax.plot(np.linspace(new_xr1, _r, 10000), link,
                        c=_color, linewidth=linewidth, rasterized=True)
                link = -np.sqrt(r[1]**2 -
                                np.linspace(_r, new_xr2, 10000)**2)
                # print(link)
                # print(dict(zip(np.linspace(_xr1, _r, 10000), link)))
                ax.plot(np.linspace(_r, new_xr2, 10000), link,
                        c=_color, linewidth=linewidth, rasterized=True)

            else:
                _r = r[1]
                if new_xr1 > 0 or new_xr2 > 0:
                    _r = r[1]
                link = -np.sqrt(r[1]**2 - np.linspace(new_xr1, _r, 10000)**2)

                ax.plot(np.linspace(new_xr1, _r, 10000), link,
                        c=_color, linewidth=linewidth, rasterized=True)
                link = np.sqrt(r[1]**2 - np.linspace(_r, new_xr2, 10000)**2)
                ax.plot(np.linspace(_r, new_xr2, 10000), link,
                        c=_color, linewidth=linewidth, rasterized=True)

                # Calculating the x, y coordinates and rotation angles of labels and the leaf points coordinates

            if y[0] == 0 and x[0] in iv_ticks:
                # print(f'{x[0]},{y[0]}')
                leaf_loc = [x[0], y[0]]
                if leaf_loc not in check_coords:
                    check_coords.append([x[0], y[0]])
                    leaf_coords.append([new_xr0, new_yr0])
                    # test_coords.append([x[0], y[0]])
                    label_coords.append(
                        [(1.05 + offset) * new_xr0, (1.05 + offset) * new_yr0, (360 - open_angle) * x[0] / xmax])
            if y[3] == 0 and x[3] in iv_ticks:
                leaf_loc = [x[3], y[3]]
                # print(f'{x[3]},{y[3]}')
                if leaf_loc not in check_coords:
                    check_coords.append([x[3], y[3]])
                    leaf_coords.append([new_xr3, new_yr3])
                    # test_coords.append([x[3], y[3]])
                    label_coords.append(
                        [(1.05 + offset) * new_xr3, (1.05 + offset) * new_yr3, (360 - open_angle) * x[2] / xmax])
        # a = len(label_coords)
        # b = len(leaf_coords)
        # c = len(check_coords)
        # print(f'label_coords is {a}')
        # print(f'leaf_coords is {b}')
        # print(f'check_coords is {c}')
        # # if y[0] == 0:
        #     label_coords.append(
        #         [(1.05 + offset) * new_xr0, (1.05 + offset) * new_yr0, (360 - open_angle) * x[0] / xmax])
        #     leaf_coords.append([new_xr0, new_yr0])
        #     #plt.text(1.05*_xr0, 1.05*_yr0, Z2['ivl'][i],{'va': 'center'},rotation_mode='anchor', rotation=360*x[0]/xmax)
        #     i += 1
        #     # print('Label_coords')
        #     # print(label_coords)
        # if y[3] == 0:
        #     label_coords.append(
        #         [(1.05 + offset) * new_xr3, (1.05 + offset) * new_yr3, (360 - open_angle) * x[2] / xmax])
        #     leaf_coords.append([new_xr3, new_yr3])
        #     #plt.text(1.05*_xr3, 1.05*_yr3, Z2['ivl'][i],{'va': 'center'},rotation_mode='anchor', rotation=360*x[2]/xmax)
        #     i += 1
        # print(label_coords)

        # print(label_coords)
        if addlabels == True:
            assert len(Z2['ivl']) == len(label_coords), "Internal error, label numbers " + \
                str(len(Z2['ivl'])) + " and " + \
                str(len(label_coords)) + " must be equal!"
            if label_colors != None:
                assert len(Z2['ivl']) == len(label_colors), "Internal error, label numbers " + str(
                    len(Z2['ivl'])) + " and " + str(len(label_colors)) + " must be equal!"
                # Adding labels
                for (_x, _y, _rot), label in zip(label_coords, Z2['ivl']):
                    ax.text(_x, _y, label, {'va': 'center'}, rotation_mode='anchor', color=list(
                        label_colors[label].keys())[0], rotation=_rot + start_angle, fontsize=fontsize)
            else:
                for (_x, _y, _rot), label in zip(label_coords, Z2['ivl']):
                    ax.text(_x, _y, label, {
                            'va': 'center'}, rotation_mode='anchor', rotation=_rot + start_angle, fontsize=fontsize)
        if addpoints == True:
            assert len(Z2['ivl']) == len(label_coords), "Internal error, point numbers " + \
                str(len(Z2['ivl'])) + " and " + \
                str(len(label_coords)) + " must be equal!"
            if point_colors != None:
                assert len(Z2['ivl']) == len(point_colors), "Internal error, label numbers " + str(
                    len(Z2['ivl'])) + " and " + str(len(point_colors)) + " must be equal!"
                for (_x, _y), label in zip(leaf_coords, Z2['ivl']):
                    point = ax.scatter(_x, _y, color=list(
                        point_colors[label].keys())[0], s=pointsize)
                    legend_elements = cvmplot.point_legend(
                        point_colors, fontsize + 2)
                    plt.legend(handles=legend_elements,
                               loc='upper left',
                               bbox_to_anchor=(1.04, 1),
                               title=point_legend_title,
                               fontsize=fontsize + 2, title_fontsize=fontsize + 3, frameon=False)
                    plt.gca().add_artist(point)

            else:
                for (_x, _y), label in zip(leaf_coords, Z2['ivl']):
                    point = ax.scatter(_x, _y, color='g', s=pointsize)
                    plt.gca().add_artist(point)

        # developing...
        # plt.draw()
        # # Plot strip
        # num_samples = 150
        # open_angle = 30
        # num_remove = math.floor(num_samples * open_angle /(360-open_angle))

        # all_samples = num_samples+num_remove
        # all_samples

        # ax.pie(np.ones(all_samples),radius=1.3,startangle=0)
        # circle = plt.Circle((0,0),1.2, fc='white', rasterized=True)
        # plt.gca().add_patch(circle)

        # if colorlabels != None:
        #     assert len(Z2['ivl'])==len(label_coords), "Internal error, label numbers "+str(len(Z2['ivl'])) +" and "+str(len(label_coords))+" must be equal!"

        #     j=0
        #     outerrad=R*1.05+width*len(colorlabels)+space*(len(colorlabels)-1)
        #     # print(outerrad)
        #     #sort_index=np.argsort(Z2['icoord'])
        #     #print(sort_index)
        #     intervals=[]
        #     for i in range(len(label_coords)):
        #         _xl,_yl,_rotl =label_coords[i-1]
        #         _x,_y,_rot =label_coords[i]
        #         if i==len(label_coords)-1:
        #             _xr,_yr,_rotr =label_coords[0]
        #         else:
        #             _xr,_yr,_rotr =label_coords[i+1]
        #         d=((_xr-_xl)**2+(_yr-_yl)**2)**0.5
        #         intervals.append(d)
        #     colorpos=intervals#np.ones([len(label_coords)])
        #     labelnames=[]
        #     for labelname, colorlist in colorlabels.items():
        #         colorlist=np.array(colorlist)[Z2['leaves']]
        #         if j!=0:
        #             outerrad=outerrad-width-space
        #         innerrad=outerrad-width
        #         patches, texts =plt.pie(colorpos, colors=colorlist,
        #                 radius=outerrad,
        #                 counterclock=True,
        #                 startangle=label_coords[0][2]*0.5)
        #         circle=plt.Circle((0,0),innerrad, fc='white')
        #         plt.gca().add_patch(circle)
        #         labelnames.append(labelname)
        #         j+=1

        #     if colorlabels_legend!=None:
        #         for i, labelname in enumerate(labelnames):
        #             print(colorlabels_legend[labelname]["colors"])
        #             colorlines=[]
        #             for c in colorlabels_legend[labelname]["colors"]:
        #                 colorlines.append(Line2D([0], [0], color=c, lw=4))
        #             leg=plt.legend(colorlines,
        #                        colorlabels_legend[labelname]["labels"],
        #                    bbox_to_anchor=(1.5+0.3*i, 1.0),
        #                    title=labelname)
        #             plt.gca().add_artist(leg)
        # elif sample_classes!=None:
        #     assert len(Z2['ivl'])==len(label_coords), "Internal error, label numbers "+str(len(Z2['ivl'])) +" and "+str(len(label_coords))+" must be equal!"

        #     j=0
        #     outerrad=R*1.05+width*len(sample_classes)+space*(len(sample_classes)-1)
        #     print(f'outerrad: {outerrad}')
        #     #sort_index=np.argsort(Z2['icoord'])
        #     #print(sort_index)
        #     intervals=[]
        #     for i in range(len(label_coords)):
        #         _xl,_yl,_rotl =label_coords[i-1]
        #         _x,_y,_rot =label_coords[i]
        #         if i==len(label_coords)-1:
        #             _xr,_yr,_rotr =label_coords[0]
        #         else:
        #             _xr,_yr,_rotr =label_coords[i+1]
        #         d=((_xr-_xl)**2+(_yr-_yl)**2)**0.5
        #         intervals.append(d)
        #     print(f'intervals:{intervals}')
        #     print(f'label_coord:{label_coords}')
        #     colorpos=intervals#np.ones([len(label_coords)])
        #     labelnames=[]
        #     colorlabels_legend={}
        #     for labelname, colorlist in sample_classes.items():
        #         ucolors=sorted(list(np.unique(colorlist)))
        #         type_num=len(ucolors)
        #         _cmp=plt.get_cmap(colormap_list[j])
        #         _colorlist=[_cmp(ucolors.index(c)/(type_num-1)) for c in colorlist]
        #         # print(f'_colorlist0:{_colorlist}')
        #         #rearange colors based on leaf index
        #         _colorlist=np.array(_colorlist)[Z2['leaves']]
        #         # print(f'_colorlist:{_colorlist}')
        #         if j!=0:
        #             outerrad=outerrad-width-space
        #         innerrad=outerrad-width
        #         # print(outerrad, innerrad)
        #         patches, texts =plt.pie(colorpos, colors=_colorlist,
        #                 radius=outerrad,
        #                 counterclock=True,
        #                 startangle=label_coords[0][2]*0.5)
        #         circle=plt.Circle((0,0),innerrad, fc='white')
        #         plt.gca().add_patch(circle)
        #         labelnames.append(labelname)
        #         colorlabels_legend[labelname]={}
        #         colorlabels_legend[labelname]["colors"]=_cmp(np.linspace(0, 1, type_num))
        #         colorlabels_legend[labelname]["labels"]=ucolors
        #         j+=1

        #     if colorlabels_legend!=None:
        #         for i, labelname in enumerate(labelnames):
        #             print(colorlabels_legend[labelname]["colors"])
        #             colorlines=[]
        #             for c in colorlabels_legend[labelname]["colors"]:
        #                 colorlines.append(Line2D([0], [0], color=c, lw=4))
        #             leg=plt.legend(colorlines,
        #                        colorlabels_legend[labelname]["labels"],
        #                    bbox_to_anchor=(1.1, 1.0-0.3*i),
        #                    title=labelname)
        #             plt.gca().add_artist(leg)
                # breakf
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.spines.left.set_visible(False)
        ax.spines.bottom.set_visible(False)
        ax.set_rasterization_zorder(None)
        plt.xticks([])
        plt.yticks([])

        if colorlabels != None:
            maxr = R * 1.05 + width * \
                len(colorlabels) + space * (len(colorlabels) - 1)
        elif sample_classes != None:
            maxr = R * 1.05 + width * \
                len(sample_classes) + space * (len(sample_classes) - 1)
        else:
            maxr = R * 1.05
        plt.xlim(-maxr, maxr)
        plt.ylim(-maxr, maxr)
        # plt.legend(loc="upper right")
        plt.subplots_adjust(left=0.05, right=0.85)
        plt.show()
        return ax

    def heatmap(data,
                order: Optional[List]=None,
                figsize=None,
                cmap=None,
                yticklabel: bool=True,
                cbar: bool=False,
                vmin: float=0,
                vmax: float=100,
                center=None,
                ax=None):
        """
        Drawing a heatmap that could concatenate to dendrogram.
        Parameters
        -----------
        data : rectangular dataset
            2D dataset that can be coerced into an ndarray. If a Pandas DataFrame
            is provided, the index/column information will be used to label the
            columns and rows.
        order: list
            A list that reindex the input 2D dataset.
        vmin, vmax : floats, optional
            Values to anchor the colormap, otherwise they are inferred from the
            data and other keyword arguments.
        cmap : matplotlib colormap name or object, or list of colors, optional
            The mapping from data values to color space. If not provided, the
            default will depend on whether ``center`` is set.
        yticklabel : bool, optional
            Whether to draw yticklabels.
        cbar : bool, optional
            Whether to draw a colorbar.
        center : float, optional
            The value at which to center the colormap when plotting divergant data.
            Using this parameter will change the default ``cmap`` if none is
            specified.
        ax : matplotlib Axes, optional
            Axes in which to draw the plot, otherwise use the currently-active Axes.

        """
        # We always want to have a DataFrame with semantic information
        # and an ndarray to pass to matplotlib

        if figsize == None:
            figsize = (10, 6)
        else:
            figsize = figsize

        if ax is None:
            ax = plt.gca()

        # process dataframe or ndarray
        if isinstance(data, pd.DataFrame):
            plot_data = data.values
        else:
            plot_data = np.asarray(data)
            data = pd.DataFrame(plot_data)

        # reindex the data frame using the list returned by scipy.cluster.hierarchy.dendrogram
        if order is not None:
            data = data.reindex(order)

        # process colormap
         # Choose default colormaps if not provided
        if cmap is None:
            if center is None:
                cmap = mpl.cm.get_cmap('rocket')
            else:
                cmap = mpl.cm.get_cmap('icefire')
        elif isinstance(cmap, str):
            cmap = get_colormap(cmap)
        elif isinstance(cmap, list):
            cmap = mpl.colors.ListedColormap(cmap)
        else:
            cmap = cmap

        # Recenter a divergent colormap
        if center is not None:

            # Copy bad values
            # in mpl<3.2 only masked values are honored with "bad" color spec
            # (see https://github.com/matplotlib/matplotlib/pull/14257)
            bad = cmap(np.ma.masked_invalid([np.nan]))[0]

            # under/over values are set for sure when cmap extremes
            # do not map to the same color as +-inf
            # under = cmap(-np.inf)
            # over = cmap(np.inf)
            under = '#c8c8c8'
            over = '#c8c8c8'
            under_set = under != cmap(0)
            over_set = over != cmap(cmap.N - 1)

            vrange = max(vmax - center, center - vmin)
            normlize = mpl.colors.Normalize(center - vrange, center + vrange)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            cmap = mpl.colors.ListedColormap(cmap(cc))
            cmap.set_bad(bad)
            if under_set:
                cmap.set_under(under)
            if over_set:
                cmap.set_over(over)
        else:
            normlize = mpl.colors.Normalize(vmin, vmax)
            cmin, cmax = normlize([vmin, vmax])
            cc = np.linspace(cmin, cmax, 256)
            cmap = mpl.colors.ListedColormap(cmap(cc))
            under = '#c8c8c8'
            over = '#c8c8c8'
            cmap.set_under(under)
            cmap.set_over(over)

        # get the data shape
        shape = list(data.shape)
        # print(data)

        xticklabels = list(data.columns)
        yticklabels = list(data.index)

        num_xlabels = len(xticklabels)
        num_ylabels = len(yticklabels)
        # print(f'num_ylabels is {num_ylabels}')

        # init a fig and ax
        # fig, ax = plt.subplots(1,1,figsize=figsize)

        # set pcolormesh x and y
        x = np.arange(num_xlabels + 1)
        y = np.arange(0, num_ylabels * 10 + 10, 10)
        # print(x)
        # print(y)
        if cbar == True:
            heatmap = ax.pcolormesh(
                x, y, plot_data, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='white')
            # axins = inset_axes(ax,
            #                    width="5%",
            #                    height="100%",
            #                    loc='upper left',
            #                    borderpad=0,
            #                    bbox_to_anchor=(1.2, 0., 1, 1),
            #                    bbox_transform=ax.transAxes,
            #                   )
            axins = inset_axes(ax,
                               width="100%",
                               height="5%",
                               loc='lower center',
                               borderpad=-3
                               # bbox_to_anchor=(1.2, 0., 1, 1),
                               # bbox_transform=ax.transAxes,
                               )
            plt.colorbar(heatmap, cax=axins, orientation="horizontal")
        else:
            heatmap = ax.pcolormesh(
                x, y, plot_data, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor='white')

        # adjust the axes and set x,y lim
        ax.set(xlim=(0, data.shape[1]), ylim=(0, data.shape[0] * 10))

        xticks, xticklabels = cvmplot._auto_ticks(ax, xticklabels, 0, shape)
        yticks, yticklabels = cvmplot._auto_ticks(ax, yticklabels, 1, shape)
        ax.set(xticks=xticks, yticks=yticks)
        print(yticks)
        print(yticklabels)
        ax.yaxis.tick_right()
        ax.set_xticklabels(xticklabels)
        if yticklabel:
            ax.set_yticklabels(yticklabels)
        else:
            ax.tick_params(axis='y', right=False, labelright=False)
        return ax

    @staticmethod
    def rotate_point(x: float, y: float, theta: float):
        """
        rotate the given point a angle based on origin (0,0)
        Parameters
        ----------
        x: float
            The x value
        y: float
            The y value
        theta: float
            The rotate angle, range is (0, 360).
        """
        angle = 2 * np.pi * theta / 360

        new_x = x * np.cos(angle) - y * np.sin(angle)
        new_y = x * np.sin(angle) + y * np.cos(angle)
        return new_x, new_y

    @staticmethod
    def point_legend(point_colors: Optional[dict]=None,
                     markersize: float=10):
        """
        Return legend elements.
        Parameters
        ----------
        point_colors: dict
            A dictionary to set the color of the leaf point. The Key is the name of the leaflabel.
            The value is the hex color.
            e.g., {'label1':{'#ffffff':'Category1'}, 'label2':{'##f77124':'Catogory2'}...}
        markersize: float
            A float to specify the markerpoint size.
        Returns
        -------
        Raises
        ------
        Notes
        -----
        References
        ----------
        See Also
        --------
        Examples
        --------
        """
        df = pd.DataFrame.from_dict(point_colors, orient='columns')
        new_df = df.melt(var_name='Labels', value_name='Cate',
                         ignore_index=False)
        new_df.index.name = 'Color'
        new_df.reset_index(inplace=True)
        new_df.dropna(inplace=True)
        new_df.drop_duplicates(subset=['Color', 'Cate'], inplace=True)
        # print(new_df)
        Cate_dict = dict(zip(new_df['Cate'], new_df['Color']))
        legend_element = [Line2D([0], [0], marker='o', color='w', markerfacecolor=Cate_dict[key],
                                 label=key, markersize=markersize) for key in Cate_dict.keys()]
        return legend_element

    @staticmethod
    def get_diff_matrix(array,
                        chunks: Optional[Tuple]=None):
        """
        Function to count the number of differences of values between rows, default ignoring NaN
        Parameters
        ----------
        array: numpy.array
            The array format of the input table
        chunks: (x, y)
            1D tuple-like of floats to specify the chunks size

        Returns
        -------
        A matrix store the number of differenct values between rows.

        Raises
        ------
        Notes
        -----
        References
        ----------
        See Also
        --------
        Examples
        --------
        """
        darray = da.from_array(array, chunks=(100, 100))
        valid_mask = da.logical_and(
            ~da.isnan(darray[:, None]), ~da.isnan(darray))
        diff_count = da.sum(valid_mask, axis=-1) - \
            da.sum(da.equal(darray[:, None], darray), axis=-1)
        diff_matrix = diff_count.compute()
        return diff_matrix

    @staticmethod
    def get_diff_df(df):
        """
        Function to count the number of differences of values between rows, default ignoring NaN
        The index of input dataframe should be your sample name.

        Parameters
        -------------
        df: pandas.dataframe
            The data frame store the MLST/cgMLST or other data

        Returns
        -------
        A dataframe store the number of differenct values between rows with sample name as the dataframe columns or index.

        Raises
        ------
        Notes
        -----
        References
        ----------
        See Also
        --------
        Examples
        --------
        """
        df = df.astype('float')
        labels = list(df.index)
        matrix = df.values
        diff_matrix = cvmplot.get_diff_matrix(matrix)
        diff_df = pd.DataFrame(diff_matrix, index=labels, columns=labels)
        return diff_df



    @staticmethod
    def _skip_ticks(labels, tickevery, startpoint, axis):
        """Return ticks and labels at evenly spaced intervals."""
        n = len(labels)
        if axis == 0:
            startpoint = 0
            if tickevery == 0:
                ticks, labels = [], []
            elif tickevery == 1:
                ticks, labels = np.arange(n) + .5 + startpoint, labels
            else:
                start_tick, end_tick, step_tick = startpoint, startpoint + n, tickevery
                ticks = np.arange(start_tick, end_tick, end_tick) + .5
                start_label, end_label, setp_label = 0, n, tickevery
                labels = labels[start_label:end_label:setp_label]
        else:
            if tickevery == 0:
                ticks, labels = [], []
            elif tickevery == 1:
                ticks, labels = np.arange(
                    0, n * 10, 10) + 5 + startpoint, labels
            else:
                start_tick, end_tick, step_tick = startpoint, startpoint + n * 10, tickevery
                ticks = np.arange(start_tick, end_tick, step_tick) + 5
                start_label, end_label, setp_label = 0, n, tickevery
                labels = labels[start_label:end_label:setp_label]
        return ticks, labels

    def get_colormap(name):
        """Handle changes to matplotlib colormap interface in 3.6."""
        try:
            return mpl.colormaps[name]
        except AttributeError:
            return mpl.cm.get_cmap(name)
