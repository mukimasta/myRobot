from myRobot import myRobot


path = "YOLO/yolov8l-worldv2.pt"

bot = myRobot()

bot.set_detector(path)
hits = bot.detect_objects()
hits.show()

id = input("Enter the id of the object you want to pick up: ")

bot.pick_object(int(id))

bot.move_to([bot.translations_list, bot.rotations_list])

input()

bot.grasp = False

