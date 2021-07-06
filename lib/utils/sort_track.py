import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy
from .sort import Sort, KalmanBoxTracker, KalmanBoxTracker1

class Tracker(object):
  def __init__(self, opt, rematch,max_age=1, min_hits=3, iou_threshold=0.3):
    self.opt = opt
    self.rematch=rematch
    self.reset()
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.iou_threshold = iou_threshold

  def init_track(self, results):
    self.frame_count = 1
    for item in results:
      if item['score'] > self.opt.new_thresh:
        self.id_count += 1
        # active and age are never used in the paper
        item['active'] = 1
        item['age'] = 1
        item['tracking_id'] = self.id_count
        if not ('ct' in item):
          bbox = item['bbox']
          item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
        self.tracks.append(item)
        trk=KalmanBoxTracker1(item)
        self.trackers_sort(trk)
        
  def reset(self):
    self.id_count = 0
    self.frame_count = 0
    self.tracks = []
    self.trackers_sort = []
    if self.rematch:
        self.rematch_ids=[]

  def step(self, results, public_det=None):
    N = len(results)
    M = len(self.tracks)
    self.frame_count += 1
    
    dets = np.array(
      [det['ct'] + det['tracking'] for det in results], np.float32) # N x 2
    track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
      (track['bbox'][3] - track['bbox'][1])) \
      for track in self.tracks], np.float32) # M
    track_cat = np.array([track['class'] for track in self.tracks], np.int32) # M
    item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
      (item['bbox'][3] - item['bbox'][1])) \
      for item in results], np.float32) # N
    item_cat = np.array([item['class'] for item in results], np.int32) # N
    tracks = np.array(
      [pre_det['ct'] for pre_det in self.tracks], np.float32) # M x 2
    dist = (((tracks.reshape(1, -1, 2) - \
              dets.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # N x M
    invalid = ((dist > track_size.reshape(1, M)) +\
      (dist > item_size.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    if M > 1:
        cells_dist=(((tracks.reshape(1, -1, 2) - \
                  tracks.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # M x M
        cells_dist=np.median(cells_dist,axis=1)
        #cells_dist=np.sort(cells_dist,axis=1)
        invalid += (dist > 1.5*cells_dist.reshape(1, M))
    dist = dist + invalid * 1e18
    
    # SORT
    if N>0:
        dets_sort=np.array([np.concatenate([result['bbox'],[result['score']]]) for result in results])# N x 5
    if N==0:
        dets_sort=np.empty((0, 5))
    
    # get predicted locations from existing trackers.
    trks = np.zeros((M, 5))
    to_del = []
    ret_sort = []
    for t, trk in enumerate(trks):
      pos = self.trackers_sort[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if np.any(np.isnan(pos)):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    trks_center = (trks[:,(0,1)]+trks[:,(2,3)])/2
    dets_sort = (dets_sort[:,(0,1)]+dets_sort[:,(2,3)])/2
    dists_sort = (((trks_center.reshape(1, -1, 2) - dets_sort.reshape(-1, 1, 2)) ** 2).sum(axis=2))
    invalid_sort = ((dists_sort > track_size.reshape(1, M)) +\
      (dists_sort > item_size.reshape(N, 1)) + \
      (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
    if M > 1:
        cells_dist=(((dets_sort.reshape(1, -1, 2) - \
          dets_sort.reshape(-1, 1, 2)) ** 2).sum(axis=2)) # M x M
        cells_dist=np.median(cells_dist,axis=1)
        invalid_sort += (dists_sort > 1.5*cells_dist.reshape(N,1))
    dists_sort = dists_sort + invalid_sort * 1e18
    
    #dist=np.minimum(dist,dists_sort)
    dist=dists_sort
    
    if self.opt.hungarian:
      item_score = np.array([item['score'] for item in results], np.float32) # N
      dist[dist > 1e18] = 1e18
      matched_indices = linear_assignment(dist)
    else:
      matched_indices = greedy_assignment(copy.deepcopy(dist))
    unmatched_dets = [d for d in range(dets.shape[0]) \
      if not (d in matched_indices[:, 0])]
    unmatched_tracks = [d for d in range(tracks.shape[0]) \
      if not (d in matched_indices[:, 1])]
    
    if self.opt.hungarian:
      matches = []
      for m in matched_indices:
        if dist[m[0], m[1]] > 1e16:
          unmatched_dets.append(m[0])
          unmatched_tracks.append(m[1])
        else:
          matches.append(m)
      matches = np.array(matches).reshape(-1, 2)
    else:
      matches = matched_indices

    ret = []
    matched_ids=[]
    for m in matches:
      track = results[m[0]]
      track['tracking_id'] = self.tracks[m[1]]['tracking_id']
      track['age'] = 1
      track['active'] = self.tracks[m[1]]['active'] + 1
      self.trackers_sort[m[1]].update(track)
      ret.append(track)
      matched_ids.append(track['tracking_id'])
      ret_sort.append(self.trackers_sort[m[1]])
          
    if self.rematch:
        for old in self.tracks:
          if old['active']==1 and not (old['tracking_id'] in matched_ids):
            self.rematch_ids.append(old['tracking_id'])

    if self.opt.public_det and len(unmatched_dets) > 0:
      # Public detection: only create tracks from provided detections
      pub_dets = np.array([d['ct'] for d in public_det], np.float32)
      dist3 = ((dets.reshape(-1, 1, 2) - pub_dets.reshape(1, -1, 2)) ** 2).sum(
        axis=2)
      matched_dets = [d for d in range(dets.shape[0]) \
        if not (d in unmatched_dets)]
      dist3[matched_dets] = 1e18
      for j in range(len(pub_dets)):
        i = dist3[:, j].argmin()
        if dist3[i, j] < item_size[i]:
          dist3[i, :] = 1e18
          track = results[i]
          if track['score'] > self.opt.new_thresh:
            self.id_count += 1
            track['tracking_id'] = self.id_count
            track['age'] = 1
            track['active'] = 1
            ret.append(track)
    else:
      # Private detection: create tracks for all un-matched detections
      for i in unmatched_dets:
        track = results[i]
        if track['score'] > self.opt.new_thresh:
          if self.rematch and len(self.rematch_ids)>0:
            track['tracking_id']=self.rematch_ids[0]
            self.rematch_ids.pop(0)
          else:
            self.id_count += 1
            track['tracking_id'] = self.id_count
          track['age'] = 1
          track['active'] =  1
          ret.append(track)
          trk = KalmanBoxTracker1(track)
          self.trackers_sort.append(trk)
          ret_sort.append(trk)
    
    result_sort=[] 
    i = len(self.trackers_sort)
    for trk in ret_sort:
        d = trk.get_state()[0]
        if (trk.time_since_update < 1):
          result_sort.append(np.concatenate((d,[trk.id])).reshape(1,-1))
    
    # Never used
    for i in unmatched_tracks:
      track = self.tracks[i]
      if track['age'] < self.opt.max_age:
        track['age'] += 1
        track['active'] = 1 # 0
        bbox = track['bbox']
        ct = track['ct']
        track['bbox'] = [
          bbox[0] + v[0], bbox[1] + v[1],
          bbox[2] + v[0], bbox[3] + v[1]]
        track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
        ret.append(track)
    self.tracks = ret
    self.trackers_sort = ret_sort
    return ret,result_sort

def greedy_assignment(dist):
  matched_indices = []
  if dist.shape[1] == 0:
    return np.array(matched_indices, np.int32).reshape(-1, 2)
  for i in range(dist.shape[0]):
    j = dist[i].argmin()
    if dist[i][j] < 1e16:
      dist[:, j] = 1e18
      matched_indices.append([i, j])
  return np.array(matched_indices, np.int32).reshape(-1, 2)