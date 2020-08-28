class STNode():
    """Sum tree Node. Stores a value equal to the sum of its children."""

    def __init__(self, idx, val=0,):
        self.idx = idx
        self.left = 2*idx + 1
        self.right = self.left + 1
        self.parent = (idx - 1)//2
        self.val = val

    def __repr__(self):
        return "STNode(idx: {}, val: {})".format(self.idx, self.val)


class SumTree():
    """Creates a sum tree stored in a list."""

    def __init__(self, capacity):
        """SumTree initialisation."""
        self.capacity = capacity
        self.tree_len = 2*capacity -1
        self.tree_array = [STNode(i) for i in range(self.tree_len)]
        self.input_pointer = self.capacity - 1

    def add(self, new_val):
        """Adds new value to sum tree. When full the oldest leaf is replaced."""
        node = self.tree_array[self.input_pointer]

        change = new_val - node.val
        node.val = new_val

        self.input_pointer += 1
        if self.input_pointer == self.tree_len:
            self.input_pointer = self.capacity - 1

        self.propagate(node.parent, change)

    def update(self, idx, new_val):
        """Update value of specific leaf node."""
        node = self.tree_array[idx]

        change = new_val - node.val
        node.val = new_val

        self.propagate(node.parent, change)

    def propagate(self, idx, change):
        """Propagates value changes to the root."""
        node = self.tree_array[idx]
        node.val += change
        if node.idx != 0:
            self.propagate(node.parent, change)

    def sample(self, s, idx=0):
        """Returns index and value corresponding to an input s."""
        node = self.tree_array[idx]

        if node.left >= self.tree_len:
            return idx, node.val

        left_sum = self.tree_array[node.left].val

        if s < left_sum:
            return self.sample(s, node.left)
        else:
            return self.sample(s-left_sum, node.right)

    def total(self):
        """Returns total sum of the tree."""
        return self.tree_array[0].val
