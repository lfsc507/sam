import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tellurium as te
import roadrunner
import re
from scipy.signal import argrelextrema as extrema
from matplotlib import animation, rc
from scipy.interpolate import griddata
from sys import exit
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as pPolygon
import matplotlib as mpl
from shapely.geometry import MultiPoint, Point, Polygon
from scipy.spatial import Voronoi
from sys import exit
from scipy.ndimage.morphology import binary_erosion
from scipy.spatial import Voronoi
from skimage import draw
from sklearn.neighbors import KDTree
import pandas as pd

def get_neib_mx(points, cutoff=55):
    '''
    points: (n, 2) numpy array for coordinates
    Return adjacency matrix
    '''
    neib_mx = np.zeros((len(points), len(points)))
    for i, pt1 in enumerate(points):
        for j in range(i+1, len(points)):
            pt2 = points[j]
            if sum(((pt1 - pt2) ** 2)) < cutoff ** 2:
                neib_mx[i,j] = 1
                neib_mx[j,i] = 1
    return neib_mx

def gen_2d_dome(n_layers=6):
    #T = [1, 10, 20, 30, 40, 50, 60]
    #R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    T = [1] + [(x+1)*10 for x in range(n_layers-1)]
    R = [0.0] + [(x+1)*0.1 for x in range(n_layers-1)]

    T = [int(np.ceil(x*0.6)) for x in T]
    #print(T)
    R = [x*20 for x in R]

    def rtpairs(r, n):
        for i in range(len(r)):
            for j in range(n[i]):
                yield r[i]*1, j*(2 * np.pi / n[i])
    points, ppoints = [], []
    for r, t in rtpairs(R, T):
        x, y = r * np.cos(t), r * np.sin(t)
        if y>=-0:
            ppoints.append([r,t])
            points.append([x,y])
    points = np.array([[x/12*255+255,y/12*255+255] for x, y in points])
    #print(len(points))
    #plt.scatter(points[:,0], points[:,1])
    #plt.show()
    #exit()

    return points, ppoints

def polygonize_by_nearest_neighbor(pp):
    """Takes a set of xy coordinates pp Numpy array(n,2) and reorders the array to make
    a polygon using a nearest neighbor approach.

    """

    # start with first index
    pp_new = np.zeros_like(pp)
    pp_new[0] = pp[0]
    p_current_idx = 0

    tree = KDTree(pp)

    for i in range(len(pp) - 1):

        nearest_dist, nearest_idx = tree.query([pp[p_current_idx]], k=4)  # k1 = identity
        nearest_idx = nearest_idx[0]

        # finds next nearest point along the contour and adds it
        for min_idx in nearest_idx[1:]:  # skip the first point (will be zero for same pixel)
            if not pp[min_idx].tolist() in pp_new.tolist():  # make sure it's not already in the list
                pp_new[i + 1] = pp[min_idx]
                p_current_idx = min_idx
                break

    pp_new[-1] = pp[0]
    return pp_new

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def gen_polys(points, ppoints):
    #generates a circular mask
    side_len = 500
    rad = 235
    mask = np.zeros(shape=(side_len, side_len))
    rr, cc = draw.circle(side_len/2, side_len/2+5, radius=rad, shape=mask.shape)
    mask[rr, cc] = 1
    mask[:int(side_len/2)-1,:] = 0

    #makes a polygon from the mask perimeter
    #se = get_circular_se(radius=1)
    #contour = mask - binary_erosion(mask, structure=se)
    contour = mask - binary_erosion(mask)
    pixels_mask = np.array(np.where(contour==1)[::-1]).T
    polygon = polygonize_by_nearest_neighbor(pixels_mask)
    polygon = Polygon(polygon)
    new_points, new_ppoints = [], []
    for i, point in enumerate(points):
        if polygon.contains(Point(point)):
            new_points.append([p*(1+np.random.uniform(-0.00,0.00)) for p in point])
            new_ppoints.append(ppoints[i])
    new_points = np.array(new_points)
    new_ppoints = np.array(new_ppoints)
    #performs voronoi tesselation
    #if len(points) > 3: #otherwise the tesselation won't work

    vor = Voronoi(new_points)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    #clips tesselation to the mask
    new_vertices = []
    for region in regions:
        poly_reg = vertices[region]
        shape = list(poly_reg.shape)
        shape[0] += 1
        p = Polygon(np.append(poly_reg, poly_reg[0]).reshape(*shape)).intersection(polygon)
        poly = (np.array(p.exterior.coords)).tolist()
        new_vertices.append(poly)

    return new_vertices

class Model2d:
    def __init__(self):
        self.points = None
        self.polys = None
        self.neib_mx = None
        self.sp_pts = {}    # Dict of special points
        self.model_str = ''
        self.r = None       # te object
        return None

    def init_2d_dome_model(self, neib_cut=55):
        self.points, ppoints = gen_2d_dome()
        self.neib_mx = get_neib_mx(self.points, cutoff=neib_cut)
        self.polys = gen_polys(self.points, ppoints)

    def add_sp_pts(self, ids=[], name=None):
        self.sp_pts[name] = ids
        return 0

    def add_basal_str(self, model_str=''):
        self.model_str = model_str

    def expand_model(self):
        n = self.points.shape[0] - 1
        model_str = self.model_str
        ms = re.finditer(r"J(\d+)", model_str)
        idj_max = max([int(m.group(1)) for m in ms])
        #print("Max J: ", idj_max)
        ms = re.finditer(r"\bW(\d+)\b", model_str)
        try: 
            idx_max = max([int(m.group(1)) for m in ms])
        except:
            idx_max = 0
        #print("Max X: ", idx_max)
        xnames = ['W'+str((idx_max+i+1)) for i in range(n-idx_max)]
        x_base = xnames[0]
        #print("X names: ", xnames)
        
        reacts = []
        
        model_lines = [x.strip() for x in model_str.split('\n')]
        for li, line in enumerate(model_lines):
            if re.match(r'J', line):
                #print(line)
                reacts.append(line)
                m = re.search(r"W(\d+)?", line)
                if m:
                    x_base = m.group(0)
                    if xnames[0] != x_base:
                        xnames.insert(0,x_base)
    #             model_lines[li] += add_epi_terms(line, xnames, x_base, mx=make_neighbor_mx(n))
        #print(x_base)
        #print(reacts)
        #print(model_lines)
        
        jid = idj_max + 1
        for xname in xnames[1:]:
            for ir, react in enumerate(reacts):
                jname = "J" + str(jid).zfill(2)
                #model_lines.append('\n'+jname+':')
                new0 = re.sub(r"J\d+", jname, react)
                #print(new0)
                new1 = re.sub(x_base, xname, new0)
                #print(new1)
                new2 = re.sub(r'\bC\d+', 'C'+xname[1:], new1)
                new3 = re.sub(r'\bL\d+', 'L'+xname[1:], new2)
                new4 = re.sub(r'\bw\d+', 'w'+xname[1:], new3)
                new5 = re.sub(r'\bc\d+', 'c'+xname[1:], new4)
                new6 = re.sub(r'\bH\d+', 'H'+xname[1:], new5)
                #print(new2)
                #new2 += add_epi_terms(new2, xnames, xname, mx=make_neighbor_mx(n))
                #print(new2)
                model_lines.append(new6)
                jid +=1
            
        for line in model_lines:
            #print(line)
            pass
        #return "\n".join(model_lines)
        self.model_str = "\n".join(model_lines)

    def set_init(self, conds=None):
        reg = r'-> ' + r'(\w+)'
        state_vars = []
        for line in self.model_str.split('\n'):
            m = re.search(reg, line)
            if m:
                state_vars.append(m.group(1))
        self.model_str += '\n\t\n\t//Species Init:\n\t'
        #print(state_vars)
        for v in state_vars:
            for pts_name, cond in conds.items():
                pts = self.sp_pts[pts_name]
                m = re.search(r'([A-Za-z]+)(\d+)', v)
                g, ci = m.group(1), int(m.group(2))
                if ci in pts and g in cond.keys():
                    self.model_str += v + '='+str(cond[g])+'; '
                else:
                    self.model_str += v + '=0.0; '
                    continue
            #self.model_str += v + '=0.0; '
        #print(self.model_str)
        #exit()

    def add_syn2eqn(self, line, var, pts, reg_term=1):
        reg = r'-> ' + var + r'(\d+)'
        m = re.search(reg, line)
        if m and int(m.group(1)) in pts:
            id_str = m.group(1)
            #print(id_str, pts)
            #new_p = 'k0' + var + id_str
            new_p = ''
            #line_new = line + ' + ' + new_p + '*' + re.sub('xxx', id_str, str(reg_term))
            line_new = line + ' + ' + re.sub('xxx', id_str, str(reg_term))
            #print(line_new)
            return line_new, new_p
        else:
            return 0, 0

    def add_syn2model(self, var, pts_name, reg_term=1):
        if pts_name in ['all', 'All', 'ALL', '', None]:
            pts = list(range(self.points.shape[0]))
        else:
            pts = self.sp_pts[pts_name]
        #print(pts)
        #exit()
        ps = []
        for line in self.model_str.split('\n'):
            line_new, new_p = self.add_syn2eqn(line, var, pts, reg_term)
            if line_new:
                if new_p:
                    ps.append(new_p)
                self.model_str = re.sub(re.escape(line), line_new, self.model_str)
                #print(line_new)
                #exit()
        #self.model_str += '\n' + '; '.join([p + ' = 1' for p in ps])

    def add_diffu2eqn(self, model_str, var, x1, x2s, sink_bounds=[]):
        new_p = None
        reg = r'-> ' + var + str(x1) + ';'
        #print(var+str(x1))

        for line in model_str.split('\n'):
            m = re.search(reg, line)
            if m:
                if len(sink_bounds) > 0:
                    if i in sink_bounds:
                        out_cells = str(len(x2s)+1)
                        #out_cells = str(len(x2s)+1)
                        #print(x1, len(x2s), sink_bounds)
                    else:
                        out_cells = str(len(x2s))
                else:
                    out_cells = str(len(x2s))
                new_p = 'D' + var
                line_new = line + ' - ' + new_p + '*' + out_cells + '*' + var + str(x1)
                line_new = line_new + ' + ' + new_p + '*(' + '+'.join([var+str(x) for x in x2s]) + ')'
                #print(line_new)
                model_str = re.sub(re.escape(line), line_new, model_str)
                #print(model_str)
                #exit()
        return model_str, new_p

    def add_diff2model(self, var, sink_bounds=[]):
        model_str = self.model_str
        for i in range(0, len(self.neib_mx)):
            #print(i, np.where(neib_mx[i])[0])
            model_str, new_p = self.add_diffu2eqn(model_str, var, i, np.where(self.neib_mx[i])[0], sink_bounds)
        self.model_str = model_str

    def run_sim(self, conds={}, gene_names=[], Ldoses=[], if_perturb=False, cell_comp=False, t_final=100):
        if not self.r:
            r = te.loada(self.model_str)
            self.r = r
        else:
            r = self.r
        r.reset()
        #cp_L_max, cp_gc, cp_gH, cp_KCL1, cp_sc = r.L_max, r.gc, r.gH, r.KCL1, None
        #r.gc, r.gH = gc, gH
        ps_cp = {}
        for p in r.ps():
            ps_cp[p] = r[p]

        for key in conds:
            r[key] = conds[key]

        #print('\n', r.gc, r.gH)
        ms_final = []
        #Ls = np.linspace(0, 15, 15)
        glist = r.getFloatingSpeciesIds()
        #r.degc = L_max
        #for var in sels[1:]:
        #for var in r.getFloatingSpeciesIds():
            #if int(var[1:]) in peri_pts:
                #r.var = np.random.uniform(0, 0.1)
            #else:
                #r.var = 0
            #print(var, 'c'+var[1:])
            #r[var] = 0
            #r['c'+var[1:]] = np.random.uniform(0, 0.01)
            #r['C'+var[1:]] = np.random.uniform(0, 0.01)
            #if int(var[1:]) in bottom_ini_pts:
                #r['W'+var[1:]] = np.random.uniform(0.9, 0.99)*0.4*4
                #r['w'+var[1:]] = np.random.uniform(0.9, 0.99)*0.4*4
            #else:
                #r['W'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01
                #r['w'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01
            #r['W'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01
            #r['w'+var[1:]] = np.random.uniform(0.9, 0.99)*0.01

        #r.timeCourseSelections = sels
        #print(r.clearModel())
        if if_perturb == False:
            m = r.simulate (0, t_final, 100)
        else:
            m1 = r.simulate(0, 50, 250)
            #for i, g in enumerate(glist):
                #if 'c' in g or 'C' in g:
                    #r[g] = 0
                    #m1[-1, i+1] = 0
                #if ('w' in g or'W' in g) and int(g[1:]) in cz_pts:
                    #r[g] = 1
                    #m1[-1, i+1] = 1
            #m2 = r.simulate(50, 100, 250)
            #m = np.vstack((m1,m2))
        ms_final.append(m[-1])
        #r.reset()
        #r.gc, r.gH = gc, gH

        ms_final = np.array(ms_final)
        #print(ms_final.shape)

        idsL = [glist.index(x)+1 for x in glist if 'L' in x]
        idsW = [glist.index(x)+1 for x in glist if 'W' in x]
        idsC = [glist.index(x)+1 for x in glist if 'C' in x]
        idsw = [glist.index(x)+1 for x in glist if 'w' in x]
        idsc = [glist.index(x)+1 for x in glist if 'c' in x]
        idsH = [glist.index(x)+1 for x in glist if 'H' in x]
        #print(m.shape, type(m), m.colnames)
        #exit()
        if 1:
            mGenes, mrGenes = [], []
            for i, ids in enumerate([idsL, idsW, idsC, idsw, idsc, idsH]):
                mrGenes.append(m[:,ids])
                if i == 10:
                    mGene = m[:,ids]
                    mGenes.append(mGene)
                    continue
                mGene = (m[:,ids] - m[:,ids].min()) / (max(m[:,ids].max(),0.3) - m[:,ids].min())
                mGenes.append(mGene)
                #print('Total', gene_names[i], "%.2f"%m[-1,ids].sum())
            mGenes, mrGenes = np.array(mGenes), np.array(mrGenes)
            mL, mW, mC, mw, mc, mH = mGenes
            mT = m[:,0]

        if 0:
            fig, ax = plt.subplots(figsize=[20, 3])

            #for i in range(1, m.shape[1])[]:
            for i in idsC:
                ax.plot(m.T[0], m.T[i], alpha=0.5)
            ax.set_xlabel('Time', size=12)
            ax.set_ylabel('Abundance', size=12)


        for p in self.r.ps():
            self.r[p] = ps_cp[p]
        r.reset()
        #print(r['kc'])
        #exit()
        #r.L_max, r.gc, r.gH, r.KCL1, r_sc = cp_L_max, cp_gc, cp_gH, cp_KCL1, cp_sc
        #print(mGenes.shape)
        #exit()
        #mGenes = pd.DataFrame(mGenes, columns=['L', 'W', 'C', 'w', 'c', 'H'])
        return mGenes, mrGenes, m[:,0]

    def plot_cbar(self, cax, cmap='inferno'):
        cax.imshow(np.vstack((np.linspace(0, 1, 256), np.linspace(0, 1, 256))).T, cmap=cmap, 
                   aspect='auto', origin='lower')
        cax.yaxis.set_ticks_position('right')
        cax.set_xticks([])
        cax.set_yticks(np.linspace(0, 256, 6))
        cax.set_yticklabels(["%.1f"%(i/256) for i in np.linspace(0, 256, 6)])
        cax.set_ylabel('Scaled abundance')

    def plot_tps(self, data, tps, gtype, norm=[], n_tpts=5):
        n_tpts = n_tpts
        fig, axs = plt.subplots(ncols=len(data), nrows=n_tpts, figsize=[7,4])
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        cax = fig.add_axes([0.93,0.7,0.01,0.15])
        cmap = plt.get_cmap('jet')
        self.plot_cbar(cax, cmap=cmap)
        scatters = []
        tpts_ids = [int(x) for x in np.linspace(0, data[0].shape[0]-1, n_tpts)]
        gene_order = [0,3,1,4,2,5]
        if len(norm)>0:
            gene_max = norm.max(axis=(0,2,3))
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                ax = axs[i,j]
                ax.set_aspect('equal')
                ax.set_facecolor('w')
                ax.axis('off')
                if j == 10:
                    vmax = (self.r.k0WL+self.r.k0WC)*self.r.k0WW
                else:
                    vmax = max(data[gene_order][j].max(),1)
                if len(norm)>0:
                    vmin = 0
                    vmax = gene_max[gene_order][j] * 1
                else:
                    vmax = data[gene_order][j].max()
                #ax.scatter(self.points[:,0], self.points[:,1], cmap=cmap, \
                    #c=data[gene_order][j][tpts_ids[i], :], norm=mpl.colors.Normalize(vmin=0, vmax=vmax))
                for ci, poly in enumerate(self.polys):
                    scaled_level = data[gene_order][j][tpts_ids[i], ci] / vmax
                    f = ax.fill(*zip(*poly), alpha=0.99, edgecolor='k',
                            facecolor=cmap(scaled_level))

        for j, name in zip(range(axs.shape[1]), 
                           ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS']):
            axs[0,j].set_title(name, size=10)
            #axs[0,j].set_title(name+" %.2f"%mrGenes[gene_order][j,:,:].max())
            #print(name, "%.2f"%data[gene_order][j,:,:].max())
        for i in range(axs.shape[0]):
            #axs[i,0].text(0, 480, 'Time '+str(i), size=10)
            axs[i,0].text(-240, 400, 'Time '+ str(int(tps[tpts_ids][i])), size=10)

        #print(r.gc, r.gH, r.L_max, '\n')
        #fig.suptitle("L_max: %.2f  gc: %.2f  gH: %.2f " % (r.L_max, r.gc, r.gH))
        fig.suptitle(gtype)

    def plot_all_gts(self, data_all, gtypes, norm_all=True):
        n_tpts = len(data_all)
        #print(data_all[0][0].shape)
        #exit()
        #data_all = np.array(data_all)
        #fig, axs = plt.subplots(ncols=len(data_all[0][0]), nrows=n_tpts, figsize=[7,5])
        fig, axs = plt.subplots(ncols=len(data_all[0]), nrows=n_tpts, figsize=[8,5])

        fig.subplots_adjust(wspace=0.01, hspace=0.01, bottom=0.05, top=0.95)
        cax = fig.add_axes([0.93,0.7,0.01,0.21])
        cmap = plt.get_cmap('jet')
        self.plot_cbar(cax, cmap=cmap)
        gene_order = [0,3,1,4,2,5]
        gene_max = data_all.max(axis=(0,2,3))
        for i in range(axs.shape[0]):
            for j in range(axs.shape[1]):
                ax = axs[i,j]
                ax.set_aspect('equal')
                ax.set_facecolor('w')
                ax.axis('off')
                #dataij = data_all[i][1][gene_order][j][-1,:]
                dataij = data_all[i][gene_order][j][-1,:]
                #dataij = np.log(dataij+0.001)

                #dataij = np.clip(dataij, 0.3, max(0.3, dataij.max()))
                #dataij = dataij - 0.3
                #dataij = dataij / (dataij.max()+0.1)
                #print(dataij.min(), dataij.max())
                vmin = 0
                if norm_all:
                    vmax = gene_max[gene_order][j] * 1
                else:
                    vmax = max(data_all[i][gene_order][j].max(), 0.1)
                for ci, poly in enumerate(self.polys):
                    scaled_level = data_all[i][gene_order][j][-1, ci] / vmax
                    f = ax.fill(*zip(*poly), alpha=0.99, edgecolor='k', 
                            facecolor=cmap(scaled_level))

                continue
                if j == 10:
                    vmax = (self.r.k0WL+self.r.k0WC)*self.r.k0WW
                else:
                    vmax = max(dataij.max(), 0.6)
                    #vmax = 3
                #vmin=0.3
                #vmax = 1
                #vmin = 0.4
                #vmin = np.log(0.001)
                #vmax = np.log(gene_max[gene_order][j])
                ax.scatter(self.points[:,0], self.points[:,1], cmap=cmap, \
                    c=dataij, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))

        for j, name in zip(range(axs.shape[1]), 
                           ['EPFL','WUS mRNA', 'WUS Protein', 'CLV3 mRNA', 'CLV3 Protein', 'HAM-WUS Signal']):
            axs[0,j].set_title(name, size=10)
        for i in range(axs.shape[0]):
            axs[i,0].text(-370, 400, gtypes[i], size=10, va='top')
        #easy_save(fig, './figures/ss_sim.svg', fmt='svg')
        #easy_save(fig, './figures/ss_sim.png', fmt='png')

    def plot_points(self, ax=None, s=10, no_num=False):
        pts = self.points
        if ax == None:
            fig, ax = plt.subplots()
        ax.scatter(pts[:,0], pts[:,1], s=s)
        if no_num == False:
            for i, pt in enumerate(pts):
                ax.text(pt[0],pt[1], str(i), zorder=10)
        #plt.show()
        return ax

    def plot_sp_pts(self, name=None, ax=None, c='k'):
        if name and ax:
            #fig, axs = plt.subplots(figsize=(10,4), nrows=2, ncols=3)
            #fig.subplots_adjust(wspace=0.01, hspace=0.01)
            #for ax in axs.flat:
            ax.set_aspect('equal')
            ax.axis('off')
            #ax.imshow(mask,cmap='Greys_r')
            for poly in self.polys:
                ax.fill(*zip(*poly), alpha=1, edgecolor='k', facecolor='w', zorder=-10)
            if name == 'all':
                points = range(len(self.points))
            else:
                points = self.sp_pts[name]
            for i in points:
                for j, poly in enumerate(self.polys):
                    if i == j:
                        ax.fill(*zip(*poly), alpha=1, \
                                edgecolor='k', facecolor=c)
            return ax

        pts = self.points
        if name:
            ids = self.sp_pts[name]
            ax = self.plot_points(s=1)
            ax.scatter(pts[ids,0], pts[ids,1], s=100)
            ax.set_title(name, color='orange', size=12)
        else:
            for name in self.sp_pts.keys():
                ids = self.sp_pts[name]
                ax = self.plot_points(s=1)
                ax.scatter(pts[ids,0], pts[ids,1], s=100)
                ax.set_title(name, color='orange', size=17)


    def plot_neib(self):
        dim = 0
        for i in range(10):
            if i*i > len(self.points):
                dim = i
                break
        fig, axs = plt.subplots(ncols=i, nrows=i, figsize=(12, 8))
        axs = axs.flatten()
        for i in range(self.neib_mx.shape[0]):
            ax = axs[i]
            neib_pts = self.points[np.where(self.neib_mx[i])[0]]
            ax = self.plot_points(ax=ax, no_num=True)
            ax.scatter(neib_pts[:,0], neib_pts[:,1],s=10)
            ax.scatter(self.points[i,0], self.points[i,1], s=10)
            ax.set_xticks([])
            ax.set_yticks([])
        #fig.tight_layout()

    def show_movie(self, data, tps, gene_names, norm=[], frames=None, interval=150):
        nGenes = data.shape[0]
        if len(norm)>0:
            gene_max = norm.max(axis=(0,2,3))
        else:
            gene_max = data.max(axis=(1,2))
        patches = [[] for i in range(nGenes)]
        colors = []
        for coords in self.polys:
            cs = np.array(coords)
            for ps in patches:
                ps.append(pPolygon(cs, closed=True))
            colors.append([0,0,1])

        fig, axs = plt.subplots(ncols=3, nrows=2, figsize=[7,3])
        cmap = plt.get_cmap('jet')

        ps = []
        for ia, ax in enumerate(axs.T.flatten()):
            p = PatchCollection(patches[ia])
            ax.set_aspect('equal')
            #ax.set_facecolor('k')
            ax.add_collection(p)
            p.set_color(cmap(0))
            p.set_edgecolor('k')
            ax.autoscale()
            ax.axis('off')
            ps.append(p)
            ax.set_title(gene_names[ia])

            #scatter = ax.scatter(new_points[:,0], new_points[:,1], c=[], marker='o', edgecolors='white')
            #scatters.append(scatter)
        tx = axs[0,2].text(400, 600, 'Time 0', size=10)

        def init():
            for p, m in zip(ps, data):
                p.set_color(cmap(0))
                p.set_edgecolor('k')
                tx.set_text('Time 0')
            return ps, tx

        def animate(i):
            for p, m, gmax in zip(ps, data, gene_max):
                p.set_color([cmap(x/gmax) for x in m[::1][i, :]])
                p.set_edgecolor('k')
                tx.set_text("Time: %.2f" % tps[::1][i])
            return ps, tx

        if not frames:
            frames = data.shape[1]
        anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=frames, interval=interval, blit=False, )

        #anim.save('dynamic_images.mp4')
        return anim
