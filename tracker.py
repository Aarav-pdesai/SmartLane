import math

class Tracker:
    def __init__(self):
        # Store the center positions of the objects and their class IDs
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, w, h, class_id = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 35:
                    self.center_points[id] = (cx, cy, class_id)
                    objects_bbs_ids.append([x, y, w, h, id, class_id]) # made a small change here, so class_id is also returned along
                    same_object_detected = True                        # object_id
                    break

            # New object is detected we assign the ID to that object
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy, class_id)
                objects_bbs_ids.append([x, y, w, h, self.id_count, class_id])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDs not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id, class_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = (center[0], center[1], class_id)

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        return objects_bbs_ids