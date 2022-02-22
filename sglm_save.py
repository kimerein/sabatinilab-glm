from collections import defaultdict
import pickle
from os.path import exists

defaultdict = defaultdict(list)

class GLM_data():
    def __init__(self, file_dir, filename):
        self.file_dir = file_dir
        self.filename = filename
        self.data = {}
        self.data['fit_results'] = []
    def save(self, overwrite=False):
        path_to_file = self.file_dir + '/' + self.filename
        if not exists(path_to_file) or overwrite:
            file_save = open(path_to_file, 'wb')
            pickle.dump(self, file_save)
            print('SGLM file saved to: ' + path_to_file)
        else:
            print('File already exists. Set overwrite=True to overwrite.')
    def load(self):
        path_to_file = self.file_dir + '/' + self.filename
        if not exists(path_to_file):
            print('File does not exist.')
            return
        file_load = open(path_to_file, 'rb')
        self.data = pickle.load(file_load)
    
    def set_uid(self, uid):
        self.data['uid'] = uid
    def set_filename(self, filename):
        self.data['filename'] = filename
    def set_basedata(self, basedata):
        self.data['basedata'] = basedata
    def set_X_cols(self, X_cols):
        self.data['X_cols'] = X_cols
    def set_gss_info(self, folds, pholdout, pgss, gssid=None):
        self.data['gss_info'] = {
            'folds': folds,
            'pholdout': pholdout,
            'pgss': pgss,
            'gssid': gssid
        }
    def set_timeshifts(self, negorder, posorder):
        self.data['negorder'] = negorder
        self.data['posorder'] = posorder
    def append_fit_results(self,
                           response_col,
                           hyperparams,
                           glm_model=None,
                           scores=None,
                           dropped_cols=[],
                           gssids=None):

        for score_id in ['tr_witi', 'tr_noiti', 'gss_witi', 'gss_noiti', 'holdout_witi', 'holdout_noiti']:
            if score_id not in scores:
                scores[score_id] = None

        fit_result = {'response_col': response_col,
                      'hyperparams': hyperparams,
                      'glm_model_gss': glm_model,
                      'dropped_cols': dropped_cols,
                      'scores': scores,
                      'gss_mse': None,
                      'refit_mse': None,
                      'gssids': gssids}
        
        self.data['fit_results'].append(fit_result)


# set_uid(self, uid)
# set_filename(self, filename)
# set_basedata(self, basedata)
# set_X_cols(self, X_cols)
# set_gss_info(self, folds, pholdout, pgss, gssid=None)
# set_timeshifts(self, negorder, posorder)
# append_fit_results(self, response_col, drop_cols, hyperparams, glm_model=None, gss_scores=None, refit_scores=None, dropped_cols=[], gssids=None)

