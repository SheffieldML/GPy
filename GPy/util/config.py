#
# This loads the configuration
#
import ConfigParser
import os
config = ConfigParser.ConfigParser()

home = os.getenv('HOME') or os.getenv('USERPROFILE')
user_file = os.path.join(home,'.gpy_config.cfg')
default_file = os.path.join('..','gpy_config.cfg')

# 1. check if the user has a ~/.gpy_config.cfg
if os.path.isfile(user_file):
    config.read(user_file)
else:
    # 2. if not, use the default one
    path = os.path.dirname(__file__)
    config.read(os.path.join(path,default_file))
