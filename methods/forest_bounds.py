"""Continuous gain and residual certification for the released 96-tree forests.

The low-level functions are taken from the study's independent interval audit.
A parameter box encloses the 62 centered features in scikit-learn's float32
input space. Each tree supplies a conservative bound over the reachable leaves.
Search returns 1 for a certified cycle, -1 for an explicit feasible error
causing a wrong prediction, or 0 for unresolved. No models or data are loaded.
"""
import numpy as np
from numba import njit


def flatten(model):
    tree_sizes = [len(t.tree_.feature) for t in model.estimators_]
    starts = np.cumsum([0] + tree_sizes[:-1])
    lo, hi, f, threshold, values = [], [], [], [], []
    for start, estimator in zip(starts, model.estimators_):
        tree = estimator.tree_
        lo.extend(tree.children_left + start)
        hi.extend(tree.children_right + start)
        f.extend(tree.feature)
        threshold.extend(tree.threshold)
        probs = tree.value[:, 0, :]
        assert np.allclose(probs.sum(axis=1),1.,rtol=0.,atol=1e-12)
        values.extend(probs)
    return tuple(np.asarray(a) for a in (starts, lo, hi, f, threshold, values))


@njit
def corners(base, sign, g0, g1, h0, h1):
    a = base * (1+g0)
    b = base * (1+g1)
    c = sign*h0
    d = sign*h1
    return min(a,b)+min(c,d), max(a,b)+max(c,d)


@njit
def enclose(f0, signs, g0, g1, h0, h1):
    low = np.zeros(62)
    high = np.zeros(62)
    for i in range(50):
        low[i], high[i] = corners(f0[i], signs[i], g0,g1,h0,h1)
    for j in range(10):
        low[52+j], high[52+j] = corners(f0[52+j], signs[5*j],g0,g1,h0,h1)
    # Range is the maximum of four affine expressions (extreme readings in each half).
    r0 = f0[:25]
    r1 = f0[25:50]
    min0, max0 = np.min(r0), np.max(r0)
    min1, max1 = np.min(r1), np.max(r1)
    spans = (max0-min0,max1-min1,max0-min1,max1-min0)
    coeff = (0.,0.,signs[0]-signs[-1], signs[-1]-signs[0])
    for k in range(4):
        a,b = corners(spans[k],coeff[k],g0,g1,h0,h1)
        low[51] = max(low[51],a)
        high[51] = max(high[51],b)
    # std is a Euclidean norm divided by sqrt(50) on the centered subspace.
    # Triangle inequality bounds it throughout the entire parameter rectangle.
    gm, hm = (g0+g1)/2, (h0+h1)/2
    mid = (1+gm)*f0[:50] + hm*signs
    sd0 = np.sqrt(np.mean(mid**2))
    distance = (g1-g0)*f0[50]/2 + (h1-h0)*abs(signs[0])/2
    low[50], high[50] = max(0.,sd0-distance),sd0+distance
    # Widen in float64 before monotonic rounding into float32 sklearn input.
    # Includes cancellation error in std and extrema as well as affine arithmetic.
    for k in range(62):
        pad = 2e-13 * (1. + abs(low[k]) + abs(high[k]))
        low[k] = np.float32(low[k] - pad)
        high[k] = np.float32(high[k] + pad)
    return low, high


@njit
def exact_point(f0, signs, gain, error):
    z = (1+gain)*f0[:50] + error*signs
    p = np.empty(62, dtype=np.float32)
    p[:50] = z
    p[50], p[51] = np.std(z), np.max(z)-np.min(z)
    for j in range(10):
        p[52+j] = np.mean(z[j*5:j*5+5])
    return p


@njit
def point_prediction(p, starts, left, right, feat, threshold, prob):
    sums = np.zeros(3)
    for i in starts:
        while feat[i] >= 0:
            i = left[i] if p[feat[i]] <= threshold[i] else right[i]
        sums += prob[i]
    return np.argmax(sums / len(starts))


@njit
def all_correct(low,high,label,starts,left,right,feat,threshold,prob):
    a,b = (label+1)%3, (label+2)%3
    score_a, score_b = 0.,0.
    target_min, rival_a_max, rival_b_max = 0.,0.,0.
    # DFS holds at most trained depth+1 nodes at once.
    stack = np.empty(128,np.int64)
    for root in starts:
        n = 1
        stack[0] = root
        min_a,min_b = 2.,2.
        leaf_target, leaf_a, leaf_b = 1.,0.,0.
        while n:
            n -= 1
            node = stack[n]
            feature = feat[node]
            if feature < 0:
                min_a = min(min_a,prob[node,label]-prob[node,a])
                min_b = min(min_b,prob[node,label]-prob[node,b])
                leaf_target = min(leaf_target,prob[node,label])
                leaf_a = max(leaf_a,prob[node,a])
                leaf_b = max(leaf_b,prob[node,b])
            else:
                if low[feature] <= threshold[node]:
                    stack[n] = left[node]
                    n += 1
                if high[feature] > threshold[node]:
                    stack[n] = right[node]
                    n += 1
        score_a += min_a
        score_b += min_b
        target_min += leaf_target
        rival_a_max += leaf_a
        rival_b_max += leaf_b
    # Monotonic addition of nonnegative values in the model's estimator order
    # also allows us to certify a tie when the smaller class index always wins.
    target = target_min / len(starts)
    rival_a = rival_a_max / len(starts)
    rival_b = rival_b_max / len(starts)
    ok_a = score_a > 1.e-10 or target > rival_a or (target == rival_a and label < a)
    ok_b = score_b > 1.e-10 or target > rival_b or (target == rival_b and label < b)
    return ok_a and ok_b


@njit
def analyze(f0,signs,label,gmax,hmax,max_boxes,starts,left,right,feat,threshold,prob):
    # Visit negative then positive half-intervals; keep the feasible origin/corners as witnesses.
    for g in (0.,-gmax,gmax):
        for h in (0.,-hmax,hmax):
            p = exact_point(f0,signs,g,h)
            pred = point_prediction(p,starts,left,right,feat,threshold,prob)
            if pred != label:
                return -1,g,h,pred,1
    boxes = [(-gmax,gmax,-hmax,hmax)]
    used = 0
    while boxes:
        g0,g1,h0,h1 = boxes.pop()
        used += 1
        if used > max_boxes:
            return 0,0.,0.,-1,used
        low,high = enclose(f0,signs,g0,g1,h0,h1)
        if all_correct(low,high,label,starts,left,right,feat,threshold,prob):
            continue
        gm,hm = (g0+g1)/2,(h0+h1)/2
        p = exact_point(f0,signs,gm,hm)
        pred = point_prediction(p,starts,left,right,feat,threshold,prob)
        if pred != label:
            return -1,gm,hm,pred,used
        if (g1-g0)*np.max(np.abs(f0[:50])) >= h1-h0 and g0 < gm < g1:
            boxes.extend(((gm,g1,h0,h1),(g0,gm,h0,h1)))
        elif h0 < hm < h1:
            boxes.extend(((g0,g1,hm,h1),(g0,g1,h0,hm)))
        else:
            return 0,0.,0.,-1,used
    return 1,0.,0.,-1,used


def raw_transform(centered, signs, gains, errors):
    z = centered * (1+gains[:,None]) + signs[None,:]*errors[:,None]
    return np.column_stack((z,np.std(z,axis=1),np.ptp(z,axis=1),
                            z.reshape(len(z),10,5).mean(axis=2)))
