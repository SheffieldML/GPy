#
# This loads the configuration
#
import os
try:
    #Attempt Python 2 ConfigParser setup
    import ConfigParser
    config = ConfigParser.ConfigParser()
    from ConfigParser import NoOptionError
except ImportError:
    #Attempt Python 3 ConfigParser setup
    import configparser
    config = configparser.ConfigParser()
    from configparser import NoOptionError

# This is the default configuration file that always needs to be present.
default_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'defaults.cfg'))

# These files are optional
# This specifies configurations that are typically specific to the machine (it is found alongside the GPy installation).
local_file = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'installation.cfg'))

# This specifies configurations specific to the user (it is found in the user home directory)
home = os.getenv('HOME') or os.getenv('USERPROFILE') or ''
user_file = os.path.join(home,'.config','GPy', 'user.cfg')

# Read in the given files.
config.read_file(open(default_file))
config.read([local_file, user_file])

if not config:
    raise ValueError("No configuration file found at either " + user_file + " or " + local_file + " or " + default_file + ".")
