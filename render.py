"""Pygame 绘图与离线事件回放；不实现搜索、训练或移动规则。"""
import pygame

BG = (13,19,30)
PANEL = (22,31,46)
TEXT = (228,235,246)
MUTED = (143,159,183)
CYAN = (63,190,194)
BLUE = (51,94,154)
ORANGE = (245,160,80)
GREEN = (87,196,151)
GOLD = (233,196,96)
PURPLE = (188,148,255)


def number(value):
    return "--" if value is None else f"{value:g}"


class DemoApp:
    def __init__(self, grid_map, sessions):
        self.grid_map, self.sessions = grid_map, sessions
        self.screen = pygame.display.set_mode((1200,780))
        pygame.display.set_caption("GridWorld Lab | Search & Q-learning")
        self.font = pygame.font.Font(None,20)
        self.small = pygame.font.Font(None,16)
        self.title = pygame.font.Font(None,32)
        self.title.set_bold(True)
        self.mode = 0
        self.playback_speed = 6
        self.running = True
        self.reset_playback()

    @property
    def session(self):
        return self.sessions[self.mode]

    @property
    def total_events(self):
        result = self.session["result"]
        if result is None:
            return 0
        return len(result["actions"]) if self.mode == 5 else len(result.events)

    @property
    def finished(self):
        return self.current_event_index >= self.total_events

    def reset_playback(self):
        self.current_event_index = 0  # 已消费事件/动作数，不是最后事件下标。
        self.is_playing = False
        self.accumulator = 0.0
        self.discovered = set()
        self.expanded = set()
        self.current = None
        self.last_event = None
        self.distances = {}

    def advance(self):
        if self.finished:
            return
        if self.mode != 5:
            event = self.session["result"].events[self.current_event_index]
            self.last_event = event
            self.current = event.current
            self.expanded.add(event.current)
            self.discovered.add(event.current)
            self.discovered.update(event.newly_discovered)
            for update in event.distance_updates:
                self.distances[update.position] = update.new_cost
        self.current_event_index += 1
        if self.finished:
            self.is_playing = False

    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        if key == pygame.K_ESCAPE:
            self.running = False
        elif pygame.K_1 <= key <= pygame.K_6:
            self.mode = key - pygame.K_1
            self.reset_playback()
        elif key == pygame.K_r:
            self.reset_playback()
        elif key == pygame.K_n:
            self.is_playing = False
            self.accumulator = 0
            self.advance()
        elif key == pygame.K_SPACE and not self.finished:
            self.is_playing = not self.is_playing
            self.accumulator = 0
        elif key in (pygame.K_PLUS,pygame.K_EQUALS,pygame.K_KP_PLUS):
            self.playback_speed = min(60,self.playback_speed+2)
        elif key in (pygame.K_MINUS,pygame.K_KP_MINUS):
            self.playback_speed = max(1,self.playback_speed-2)

    def update(self, seconds):
        if self.is_playing:
            self.accumulator += seconds
            while self.accumulator >= 1/self.playback_speed and self.is_playing:
                self.accumulator -= 1/self.playback_speed
                self.advance()

    def text(self, value, x, y, color=TEXT, font=None):
        self.screen.blit((font or self.font).render(str(value),True,color),(x,y))

    def draw_map(self):
        m = self.grid_map
        size = min(620//m.width,450//m.height)
        left, top = 44+(700-m.width*size)//2, 180+(450-m.height*size)//2
        def center(p):
            return left+p[1]*size+size//2, top+p[0]*size+size//2
        result = self.session["result"]
        path = ()
        agent = None
        if result is not None:
            if self.mode == 5:
                path = result["trajectory"][:self.current_event_index+1]
                agent = path[-1]
            elif self.finished and result.success:
                path = result.path
        for row in range(m.height):
            for col in range(m.width):
                p = row,col
                color = (33,44,60) if m.grid[row][col] == 0 else (43,46,54)
                if p in self.discovered: color = (36,100,111)
                if p in self.expanded: color = BLUE
                if p in path: color = (112,77,48)
                rect = pygame.Rect(left+col*size+3,top+row*size+3,size-6,size-6)
                pygame.draw.rect(self.screen,color,rect,border_radius=9)
                if m.grid[row][col] == 1:
                    pygame.draw.line(self.screen,(69,73,83),rect.topleft,rect.bottomright,2)
                    pygame.draw.line(self.screen,(69,73,83),rect.topright,rect.bottomleft,2)
                if p == self.current:
                    pygame.draw.rect(self.screen,CYAN,rect,3,border_radius=9)
        if len(path)>1:
            pygame.draw.lines(self.screen,ORANGE,False,[center(p) for p in path],5)
        for p,label,color in ((m.start,"S",GREEN),(m.goal,"G",GOLD)):
            x,y = center(p)
            pygame.draw.circle(self.screen,color,(x,y),max(10,size//5))
            surface = self.font.render(label,True,BG)
            self.screen.blit(surface,surface.get_rect(center=(x,y)))
        if agent is not None:
            pygame.draw.circle(self.screen,PURPLE,center(agent),max(9,size//7))
            pygame.draw.circle(self.screen,TEXT,center(agent),max(9,size//7),2)

    def draw(self):
        self.screen.fill(BG)
        self.text("GRIDWORLD / LAB",32,22,font=self.title)
        self.text("Search, compare, replay.",34,65,MUTED)
        self.text("FIXED MAP  /  DETERMINISTIC",825,40,CYAN,font=self.small)
        pygame.draw.rect(self.screen,PANEL,(24,105,750,587),border_radius=16)
        pygame.draw.rect(self.screen,PANEL,(794,105,382,587),border_radius=16)
        self.text(self.session["name"],44,115)
        self.text("4-direction RL" if self.mode==5 else "Node-event replay, not a heap snapshot",
                  44,148,MUTED,font=self.small)
        self.draw_map()
        labels = ("BFS","Dijkstra","Greedy best-first","A* / 4 directions","A* / 8 directions","Q-learning")
        for i,label in enumerate(labels):
            y = 120+i*36
            if i==self.mode:
                pygame.draw.rect(self.screen,(37,66,82),(810,y-2,350,33),border_radius=7)
            self.text(f"{i+1}   {label}",826,y,CYAN if i==self.mode else MUTED)
        result = self.session["result"]
        if result is None:
            self.text("MODEL UNAVAILABLE",818,354,ORANGE)
            # 按像素宽度换行，错误路径不会溢出统计面板。
            words = self.session["error"].split()
            line, y = "",390
            for word in words:
                if self.small.size(line+word)[0]>332:
                    self.text(line,818,y,MUTED,font=self.small)
                    y += 23; line = ""
                # Windows长路径按字符截断，仅显示说明前部。
                line += word[:40]+" "
                if y>580: break
            self.text(line,818,y,MUTED,font=self.small)
        else:
            success = result["success"] if self.mode==5 else result.success
            self.text("RESULT  /  "+("SUCCESS" if success else "NO ROUTE / LIMIT"),818,352,
                      GREEN if success else ORANGE)
            if self.mode == 5:
                position = result["trajectory"][self.current_event_index]
                rows = [("Agent position",str(position)),
                        ("Moves replayed",f"{self.current_event_index} / {result['steps']}"),
                        ("Total return (evaluation)",number(result["total_reward"])),
                        ("Evaluation steps",str(result["steps"])),
                        ("Policy","Greedy / no training")]
            else:
                rows = [("Path cost",number(result.cost)),("Path steps",number(result.path_steps)),
                        ("Expanded (total)",str(result.expanded_count)),
                        ("Discovered (total)",str(result.discovered_count)),
                        ("Search time",f"{self.session['elapsed_ms']:.3f} ms")]
            for i,(label,value) in enumerate(rows):
                y = 391+i*35
                self.text(label,818,y,MUTED,font=self.small)
                self.text(value,1152-self.font.size(value)[0],y)
            if self.mode != 5:
                e = self.last_event
                self.text(f"Current g: {number(e.current_cost if e else None)}   h: {number(e.current_heuristic if e else None)}",
                          818,575,CYAN,font=self.small)
                updates = len(e.distance_updates) if e else 0
                self.text(f"Distance updates in event: {updates}",818,602,MUTED,font=self.small)
            else:
                self.text("Return is reward, not search cost.",818,590,MUTED,font=self.small)
        status = "UNAVAILABLE" if result is None else ("COMPLETE" if self.finished else
                 ("PLAYING" if self.is_playing else "PAUSED"))
        self.text(f"{status}   {self.current_event_index}/{self.total_events}   |   {self.playback_speed} steps/s",44,654,CYAN,font=self.small)
        self.text("1-6 Select    SPACE Play / pause    N Step    R Reset    +/- Speed    ESC Exit",32,716)
        self.text("Green: start   Gold: goal   Cyan: discovered   Blue: expanded   Orange: path   Purple: agent",32,750,MUTED,font=self.small)
        pygame.display.flip()

    def run(self):
        clock = pygame.time.Clock()
        while self.running:
            seconds = clock.tick(60)/1000
            for event in pygame.event.get():
                self.handle_event(event)
            self.update(seconds)
            self.draw()
