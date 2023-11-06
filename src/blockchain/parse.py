

class ArgParse:
    def __init__(self, argument: str) -> None:
        self.argument = argument.split()
    
    def add_argument(self, arg_item : str, arg_name : str, arg_type=None, help=None):
        if arg_item not in self.argument: 
            raise IndexError("There is no argument {}".format(arg_item))
        elif arg_item[0] != '-':
            raise KeyError("The argument item must start with \'-\'")
        elif len(arg_name)<=2 or '--' not in arg_name[0:2]:
            raise KeyError("The arg_name must start with \'--\'.")
        arg_index = self.argument.index(arg_item) + 1
        if arg_index >= len(self.argument):
            raise IndexError("Invalid argument.") 
        arg = self.argument[arg_index]
        self.__dict__[arg_name[2:]] = arg_type(arg) if arg_type else arg

