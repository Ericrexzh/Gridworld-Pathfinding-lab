# GridWorld Lab

Toy 级课程实践：BFS、Dijkstra、Greedy、A* 四/八方向与已训练 Q-learning 的统一回放。窗口 1200×780，深色网格，不依赖外部图片。界面使用英文短标签。

## 安装与运行

已验证 Python 3.10.11；依赖固定为 Gymnasium 1.4.0、NumPy 2.2.6、Pygame 2.6.1。

```powershell
python -m pip install -r requirements.txt
python main.py
```

默认地图和模型路径相对于 main.py 所在目录，不依赖启动时的工作目录。导入 main/render 不会打开窗口。首次显示为暂停，按 Space 开始。

## 操作

- 1：BFS；2：Dijkstra；3：Greedy；4：A* 四方向；5：A* 八方向；6：Q-learning。
- Space：播放/暂停；N：暂停并推进一个事件或一个 Agent 动作。
- R：清空回放游标与展示状态；不重新搜索、不重新训练。
- + / -：调整速度，范围1–60步/秒；普通键盘 = 也可加速。
- ESC 或窗口关闭按钮：退出。

算法名称列表是键盘快捷键说明，不是鼠标按钮。切换算法会暂停并重置展示。回放结束按 R 再播放。

绿色起点、金色终点、深灰障碍、青色已发现、蓝色已扩展、橙色路径、紫色RL Agent。搜索最终路径在事件回放完成后出现；RL轨迹随Agent逐步显示。

统计中的搜索结果、路径成本、发现/扩展总数在启动时已计算；底部显示当前回放进度。事件不是完整优先队列快照。RL显示当前坐标、已回放动作数和独立评估的最终累计奖励，不把访问次数称为扩展节点数。

## 算法规则

- BFS、Dijkstra、Greedy均四方向，正交代价10。
- A*四方向使用10倍曼哈顿；八方向正交10、对角14.14，使用Octile。
- 八方向禁止穿角：两个相邻正交格和目标必须可通行。
- RL仅四方向，0上、1右、2下、3左；非法移动原地，普通/无效移动奖励-1，到达终点+10。
- 窗口加载Q表并执行独立贪心评估，不执行训练。搜索成本与RL奖励含义不同。

## 文件

```text
main.py / render.py       入口、绘图、回放控制
grid.py / search.py       不可变地图与搜索算法
envs/grid_world.py        Gymnasium环境及纯模型查询
rl/                      动态规划、Q-learning、独立评估
train.py                 独立训练入口
maps/demo.json           演示地图
results/q_table.npz      已训练Q表
results/training_history.csv  3000回合训练记录
results/training_config.json  参数、地图、统计与评估
tests/                   151项回归测试
docs/                    规范与前序教学报告
```

模型缺失、损坏或维度不符时，Q-learning面板显示提示，五种搜索仍可使用。若training_config.json存在，额外核对训练地图与当前地图。仅有Q表时只能检查尺寸，不能确认地图身份。

当前保存的demo策略独立评估为8步、累计奖励3。固定地图结果不表示跨地图泛化。窗口不会更改Q表、训练记录或地图。

## 验证与可选训练

```powershell
python -X utf8 -B -m unittest discover -s tests -v
```

现有模型可直接演示，不需要重训。如以后主动重新训练，可运行 `python train.py`，它会覆盖默认results文件；可用 `--output results_new` 保存到其他目录。训练与窗口分开运行。

界面限固定地图和预计算回放，没有地图编辑、随机地图、实时训练或额外学习算法。搜索耗时仅计算法调用，不包含动画时间。
