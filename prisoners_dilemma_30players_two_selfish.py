import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, name, is_shortsighted=False):
        self.name = name
        self.q_table = {}
        self.learning_rate = 0.2
        self.discount_factor = 0.95
        self.epsilon = 0.3
        self.last_action = None
        self.last_state = None
        self.history = []
        self.memory_length = 5
        self.is_shortsighted = is_shortsighted
        
    def get_state(self, other_players_actions):
        if self.is_shortsighted:  # 단기적 시야 플레이어는 항상 직전 행동만 고려
            my_last = self.history[-1] if self.history else None
            other_last = [
                actions[-1] if actions and len(actions) > 0 else None 
                for actions in other_players_actions
            ]
            return (my_last,) + tuple(other_last)
        else:
            # 장기적 시야 사용
            my_recent = self.history[-self.memory_length:] if self.history else []
            while len(my_recent) < self.memory_length:
                my_recent.insert(0, None)
                
            other_recent = []
            for actions in other_players_actions:
                if actions is None or len(actions) == 0:
                    temp = [None] * self.memory_length
                else:
                    temp = list(actions[-self.memory_length:])
                    while len(temp) < self.memory_length:
                        temp.insert(0, None)
                other_recent.extend(tuple(temp))
                
            return tuple(my_recent) + tuple(other_recent)
        
    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {'C': 5000, 'D': 5000}
            
        if np.random.random() < self.epsilon:
            action = np.random.choice(['C', 'D'])
        else:
            action = max(self.q_table[state].items(), key=lambda x: x[1])[0]
            
        self.last_action = action
        self.last_state = state
        self.history.append(action)
        return action
        
    def learn(self, reward, new_state):
        if self.last_state is None:
            return
            
        if new_state not in self.q_table:
            self.q_table[new_state] = {'C': 5000, 'D': 5000}
            
        old_value = self.q_table[self.last_state][self.last_action]
        next_max = max(self.q_table[new_state].values())
        
        new_value = (1 - self.learning_rate) * old_value + \
                    self.learning_rate * (reward + self.discount_factor * next_max)
        
        self.q_table[self.last_state][self.last_action] = new_value

class PrisonersDilemma:
    def __init__(self):
        # 30명의 플레이어 중 2명이 단기적 시야를 가짐
        self.agents = [
            Agent(f"Player{i}", is_shortsighted=(i < 2))  # Player0과 Player1이 단기적 시야
            for i in range(30)
        ]
        self.history = []
        self.cumulative_rewards = [[0] for _ in range(30)]
        self.cooperation_history = []
        
    def get_reward(self, actions):
        cooperator_count = sum(1 for action in actions if action == 'C')
        betrayer_count = len(actions) - cooperator_count
        
        rewards = []
        for action in actions:
            if action == 'C':
                if betrayer_count == 0:  # 모두 협력
                    rewards.append(500000)  # 협력 보상
                elif betrayer_count <= 3:  # 소수만 배신
                    rewards.append(200000 - 30000 * betrayer_count)  # 기본 보상 유지
                else:  # 다수가 배신
                    rewards.append(50000)  # 최소한의 보상 보장
            else:  # action == 'D'
                if betrayer_count == 1:  # 혼자만 배신
                    rewards.append(600000)  # 배신 이득
                elif betrayer_count <= 3:  # 소수가 배신
                    rewards.append(100000)  # 소수 배신 이득
                elif betrayer_count <= 10:  # 중간 수준 배신
                    rewards.append(30000)  # 약간의 이득
                else:  # 다수가 배신
                    rewards.append(10000)  # 최소 보상
        return rewards
        
    def play_game(self, episodes=1000):
        for episode in range(episodes):
            # 매 에피소드마다 플레이어 순서를 랜덤하게 섞음
            player_order = np.random.permutation(len(self.agents))
            current_actions = [None] * len(self.agents)
            
            # 각 에이전트의 행동 선택 (랜덤 순서로)
            for idx in player_order:
                agent = self.agents[idx]
                other_histories = []
                for j, other_agent in enumerate(self.agents):
                    if j != idx:
                        other_histories.append(other_agent.history)
                
                state = agent.get_state(other_histories)
                action = agent.choose_action(state)
                current_actions[idx] = action
            
            rewards = self.get_reward(current_actions)
            
            # 각 에이전트 학습 (역시 랜덤 순서로)
            for idx in np.random.permutation(len(self.agents)):
                agent = self.agents[idx]
                other_histories = []
                for j, other_agent in enumerate(self.agents):
                    if j != idx:
                        other_histories.append(other_agent.history)
                new_state = agent.get_state(other_histories)
                agent.learn(rewards[idx], new_state)
            
            # 누적 보상 업데이트
            for i in range(len(self.agents)):
                self.cumulative_rewards[i].append(
                    self.cumulative_rewards[i][-1] + rewards[i]
                )
            
            cooperation_count = sum(1 for action in current_actions if action == 'C')
            self.cooperation_history.append(cooperation_count / len(self.agents))
            
            # epsilon 감소
            if episode % 100 == 0:
                for agent in self.agents:
                    agent.epsilon = max(0.05, agent.epsilon * 0.995)
            
            self.history.append((current_actions, rewards))
            
            # 진행 상황 출력
            if episode % 1000 == 0:
                print(f"Episode {episode}")
                print(f"전체 협력 비율: {np.mean(self.cooperation_history[-100:]):.2%}")
                print(f"단기적 시야 플레이어들의 최근 협력 비율:")
                for i in range(2):
                    print(f"Player{i}: {sum(1 for a in self.agents[i].history[-100:] if a == 'C')/100:.2%}")
                print("---")

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # 누적 보상 그래프
        # 단기적 시야 플레이어들은 굵게 표시
        for i in range(2):
            ax1.plot(self.cumulative_rewards[i], 
                    label=f'Player{i} (단기적 시야)', 
                    linewidth=3)
        
        # 장기적 시야 플레이어들의 평균 선
        avg_rewards = np.mean([self.cumulative_rewards[i] for i in range(2, len(self.agents))], axis=0)
        ax1.plot(avg_rewards, 
                label='장기적 시야 플레이어들 평균', 
                linewidth=2,
                color='blue')
        
        # 나머지 플레이어들은 투명하게
        for i in range(2, len(self.agents)):
            ax1.plot(self.cumulative_rewards[i], 
                    alpha=0.1, 
                    color='gray',
                    linewidth=1)
        
        ax1.set_title('누적 보상 변화')
        ax1.set_xlabel('에피소드')
        ax1.set_ylabel('누적 보상 (원)')
        ax1.legend()
        ax1.grid(True)
        
        # 협력 비율 그래프
        window_size = 100
        moving_avg = [sum(self.cooperation_history[i:i+window_size])/window_size 
                     for i in range(len(self.cooperation_history)-window_size+1)]
        
        ax2.plot(range(window_size-1, len(self.cooperation_history)), moving_avg,
                label='전체 평균', linewidth=2)
        
        # 단기적 시야 플레이어들의 협력 비율
        colors = ['red', 'orange']
        for i in range(2):
            player_history = [1 if h[0][i] == 'C' else 0 for h in self.history]
            moving_avg_player = [sum(player_history[i:i+window_size])/window_size 
                               for i in range(len(player_history)-window_size+1)]
            ax2.plot(range(window_size-1, len(player_history)), moving_avg_player,
                    label=f'Player{i} (단기적 시야)', 
                    linestyle='--', 
                    linewidth=2, 
                    color=colors[i])
        
        ax2.set_title('협력 비율 변화 (이동 평균)')
        ax2.set_xlabel('에피소드')
        ax2.set_ylabel('협력 비율')
        ax2.set_ylim(0, 1)
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 최종 통계 출력
        print("\n=== 최종 통계 ===")
        print("단기적 시야 플레이어들:")
        for i in range(2):
            print(f"Player {i}: {self.cumulative_rewards[i][-1]:,}원")
        
        print("\n장기적 시야 플레이어들의 평균:")
        longterm_rewards = [self.cumulative_rewards[i][-1] for i in range(2, len(self.agents))]
        print(f"평균: {np.mean(longterm_rewards):,.0f}원")
        print(f"최대: {np.max(longterm_rewards):,}원")
        print(f"최소: {np.min(longterm_rewards):,}원")
        print(f"표준 편차: {np.std(longterm_rewards):,.0f}원")
        
        print(f"\n마지막 1000회 전체 평균 협력 비율: {np.mean(self.cooperation_history[-1000:]):.2%}")
        print("마지막 1000회 단기적 시야 플레이어들의 협력 비율:")
        for i in range(2):
            print(f"Player{i}: {sum(1 for h in self.history[-1000:] if h[0][i] == 'C')/1000:.2%}")

# 게임 실행 및 결과 시각화
game = PrisonersDilemma()
game.play_game(10000)
game.plot_results() 