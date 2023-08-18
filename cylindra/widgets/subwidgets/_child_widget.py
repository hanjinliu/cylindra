from magicclass import MagicTemplate


# the following is a template for child widgets of CylindraMainWidget
class ChildWidget(MagicTemplate):
    def _get_main(self):
        from cylindra.widgets import CylindraMainWidget

        return self.find_ancestor(CylindraMainWidget, cache=True)
