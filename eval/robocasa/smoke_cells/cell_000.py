# eval/robocasa replay smoke (no model): boot, lines, gripper, SAM3 locate, straight-down yaw
s = robo.state()
print("task:", s.task)
print("tcp", s.right.tcp_pos, "pitch/yaw/roll", s.right.pitch, s.right.yaw, s.right.roll, "gripper", s.right.gripper)
x, y, z = s.right.tcp_pos
m = robo.move_to([x, y, z - 0.05], arm="right", path="line")
print("move down 5 cm:", m.reached, m.ik_error, m.steps_used)
g = robo.set_gripper("right", robo.CLOSE)
print("close:", g.opening, g.holding)
g = robo.set_gripper("right", robo.OPEN)
print("open:", g.opening)
o = robo.locate("drawer handle")
print("locate drawer handle:", o.found, o.center_xyz, o.views, o.score, o.note)
for yaw in (30.0, 75.0):
    m = robo.move_to([x, y, z - 0.05], arm="right", pitch=90, yaw=yaw)
    a = robo.state().right
    print(f"straight down, yaw {yaw}:", m.reached, m.steps_used, "-> pitch/yaw/roll", round(a.pitch, 1), round(a.yaw, 1), round(a.roll, 1))
print("state step", robo.state().step)
