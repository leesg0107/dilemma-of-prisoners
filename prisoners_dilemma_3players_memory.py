import numpy as np
import matplotlib.pyplot as plt

class Agent:
    def __init__(self, name):
        self.name = name
        self.q_table = {}
        self.learning_rate = 0.2  # 학습률 증가
        self.discount_factor = 0.95
        self.epsilon = 0.3
        self.last_action = None
        self.last_state = None
        self.history = []
        self.memory_length = 5  # 더 긴 기억 사용
        
    def get_state(self, other_players_actions):
        # 최근 N회의 자신의 행동과 다른 플레이어들의 행동을 함께 고려
        my_recent = self.history[-self.memory_length:] if self.history else []
        # 기억 길이만큼 None으로 채우기
        while len(my_recent) < self.memory_length:
            my_recent.insert(0, None)
            
        # 다른 플레이어들의 행동도 같은 방식으로 처리
        other_recent = []
        for actions in other_players_actions:
            if actions is None:
                other_recent.append([None] * self.memory_length)
            else:
                recent = list(actions[-self.memory_length:])
                while len(recent) < self.memory_length:
                    recent.insert(0, None)
                other_recent.extend(recent)
                
        return tuple(my_recent + other_recent)
        
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
        self.agents = [Agent(f"Player{i}") for i in range(3)]
        self.history = []
        self.cumulative_rewards = [[0], [0], [0]]
        self.cooperation_history = []
        
    def get_reward(self, actions):
        if all(action == 'C' for action in actions):
            return [30000, 30000, 30000]
        elif all(action == 'D' for action in actions):
            return [-10000, -10000, -10000]
        
        rewards = []
        for i, action in enumerate(actions):
            if action == 'D':
                betrayer_count = sum(1 for a in actions if a == 'D')
                if betrayer_count == 1:
                    rewards.append(50000)
                elif betrayer_count == 2:
                    rewards.append(20000)
            else:  # action == 'C'
                betrayer_count = sum(1 for a in actions if a == 'D')
                if betrayer_count == 1:
                    rewards.append(-20000)
                elif betrayer_count == 2:
                    rewards.append(-30000)
        return rewards
        
    def play_game(self, episodes=1000):
        for episode in range(episodes):
            current_actions = []
            
            # 각 에이전트의 행동 선택
            for i, agent in enumerate(self.agents):
                # 다른 플레이어들의 행동 기록 수집
                other_histories = []
                for j, other_agent in enumerate(self.agents):
                    if j != i:
                        other_histories.append(other_agent.history)
                    
                state = agent.get_state(other_histories)
                action = agent.choose_action(state)
                current_actions.append(action)
            
            rewards = self.get_reward(current_actions)
            
            # 각 에이전트 학습
            for i, agent in enumerate(self.agents):
                other_histories = []
                for j, other_agent in enumerate(self.agents):
                    if j != i:
                        other_histories.append(other_agent.history)
                new_state = agent.get_state(other_histories)
                agent.learn(rewards[i], new_state)
            
            # 누적 보상 업데이트
            for i in range(3):
                self.cumulative_rewards[i].append(
                    self.cumulative_rewards[i][-1] + rewards[i]
                )
            
            cooperation_count = sum(1 for action in current_actions if action == 'C')
            self.cooperation_history.append(cooperation_count / 3)
            
            # epsilon 감소
            if episode % 100 == 0:
                for agent in self.agents:
                    agent.epsilon = max(0.05, agent.epsilon * 0.995)
            
            self.history.append((current_actions, rewards))
            
            # 진행 상황 출력
            if episode % 1000 == 0:
                print(f"Episode {episode}: 협력 비율 = {np.mean(self.cooperation_history[-100:]):.2%}")

    def plot_results(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 누적 보상 그래프
        for i in range(3):
            ax1.plot(self.cumulative_rewards[i], label=f'Player {i+1}')
        ax1.set_title('누적 보상 변화')
        ax1.set_xlabel('에피소드')
        ax1.set_ylabel('누적 보상 (원)')
        ax1.legend()
        ax1.grid(True)
        
        # 협력 비율 그래프
        window_size = 100
        moving_avg = [sum(self.cooperation_history[i:i+window_size])/window_size 
                     for i in range(len(self.cooperation_history)-window_size+1)]
        
        ax2.plot(range(window_size-1, len(self.cooperation_history)), moving_avg)
        ax2.set_title('협력 비율 변화 (이동 평균)')
        ax2.set_xlabel('에피소드')
        ax2.set_ylabel('협력 비율')
        ax2.set_ylim(0, 1)
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 최종 통계 출력
        print("\n=== 최종 통계 ===")
        for i in range(3):
            print(f"Player {i+1} 최종 누적 보상: {self.cumulative_rewards[i][-1]:,}원")
        print(f"\n마지막 1000회 평균 협력 비율: {np.mean(self.cooperation_history[-1000:]):.2%}")

# 게임 실행 및 결과 시각화
game = PrisonersDilemma()
game.play_game(10000)
game.plot_results() 