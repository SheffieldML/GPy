#
# This loads the configuration
#
import ConfigParser
import os
config = ConfigParser.ConfigParser()

home = os.getenv('HOME') or os.getenv('USERPROFILE')
user_file = os.path.join(home,'.gpy_config.cfg')
default_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'gpy_config.cfg'))
# print user_file, os.path.isfile(user_file)
# print default_file, os.path.isfile(default_file)

# 1. check if the user has a ~/.gpy_config.cfg
if os.path.isfile(user_file):
    config.read(user_file)
elif os.path.isfile(default_file):
    # 2. if not, use the default one
    config.read(default_file)
else:
    #3. panic
    raise ValueError, "no configuration file found"
