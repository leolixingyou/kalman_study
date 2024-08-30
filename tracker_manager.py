### Tracker Manager for several tracker system modeling

class Tracker_Manager:
    def __init__(self, type_tracker):
        self.manager_type_of_tracker(type_tracker)
    
    ### type_tracker: [xyxy, xywh, xy]
    def manager_type_of_tracker(self, type_tracker):
        ## modify refer format as f'box_{type_tracker}_tracker'()
        method_name = f'box_{type_tracker}_tracker'
        method_f = getattr(self, method_name, None)
        method_f()

    def box_xyxy_tracker(self):
        print('xyxy')

    def box_xywh_tracker(self):
        print('xywh')

    def box_xmym_tracker(self):
        print('xmym')


if __name__ == "__main__":
    type_tracker_list = ['xyxy', 'xywh', 'xmym']
    # type_tracker = type_tracker_list[0]
    for type_tracker in type_tracker_list:
        tracker_manager = Tracker_Manager(type_tracker)