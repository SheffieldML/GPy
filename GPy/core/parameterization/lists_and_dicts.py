'''
Created on 27 Feb 2014

@author: maxz
'''

class ParamList(list):
    """
    List to store ndarray-likes in.
    It will look for 'is' instead of calling __eq__ on each element.
    """
    def __contains__(self, other):
        for el in self:
            if el is other:
                return True
        return False

    pass
