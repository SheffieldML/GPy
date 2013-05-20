import GPy
import urllib2

# TODO...
class mocap_fetch(base_url = 'http://mocap.cs.cmu.edu:8080/subjects/', skel_store_dir = './', motion_store_dir = './'):
    def __init__(self):
        self.base_url = base_url
        self.store_dir = store_dir
        self.motion_dict = []
    
    def fetch_motions(self, motion_dict = None):
        response = urllib2.urlopen(...)
        html = response.read()
