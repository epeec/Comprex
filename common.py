class Gettable(object):
    def get(self):
        return self.obj

class Subscribable(object):
    def __init__(self):
        self.dependency_list=[]
    def __del__(self):
        for item in self.dependency_list:
            # gaspi_printf("Deleting %s from class %s"%(item, self))
            del item
    def subscribe(self, obj):
        self.dependency_list.append(obj)