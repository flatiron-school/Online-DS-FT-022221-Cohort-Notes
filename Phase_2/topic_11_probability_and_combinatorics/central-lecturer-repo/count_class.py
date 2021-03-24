class Counter:
    
    def __init__(self, arr, final_list=None):
        self.arr = arr
        if final_list == None:
            final_list = []
        self.final_list = final_list
    
    def calc(self, select, order=False):
        self.select = select
        self.order = order
        import itertools
        
        if self.order == False:
            combos = set(itertools.combinations(self.arr, self.select))
            num = len(combos)
            self.final_list = list(combos)
        else:
            perms = list(itertools.permutations(self.arr, self.select))
            self.final_list = []
            for perm in perms:
                if perm not in self.final_list:
                    self.final_list.append(perm)

        return len(self.final_list)