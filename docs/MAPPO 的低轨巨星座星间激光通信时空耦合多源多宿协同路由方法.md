
$$
\boxed{  
\text{用 MAPPO 学习动态星间激光网络中多源多宿业务流的时空占用轨迹协调。}  
}  
$$

不是普通的“多条流分别找最短路”，而是解决：多条源宿业务在动态拓扑、光终端互斥、链路容量、路径寿命和未来冲突窗口共同作用下，如何协同保持或提前切换路径。

---

# 一、Problem

传统单源单宿路由解决的是：

$$
\text{一对源宿之间怎么找一条低时延路径}  
$$

但现在要解决的是：

$$
\text{多对源宿同时通信时，所有业务流如何在动态星座网络中协同选择路径}  
$$

具体来说，系统中存在多条并发业务流：

$$
\mathcal{F}={1,2,\dots,F}  
$$

第 $f$ 条业务流对应一对源宿节点：

$$
(s_f,d_f)  
$$

在时隙 $k$，它选择一条路径：

$$
\pi_f^{(k)}  
$$

所有业务流的联合路径集合为：
$$
\mathbf{\Pi}^{(k)}

\left[  
\pi_1^{(k)},\pi_2^{(k)},\dots,\pi_F^{(k)}  
\right]  
$$

目标不是让每一条流分别最优，而是让所有流的联合路径演化最优：

$$
\min_{{\mathbf{\Pi}^{(k)}}_{k=1}^{K}}  
\sum_{k=1}^{K}  
J^{(k)}  
$$

其中 $J^{(k)}$ 是时隙 $k$ 的系统总代价，包括平均时延、峰值时延、切换代价、建链代价、互斥冲突、中断风险和未来冲突风险。

---

# 二、为什么这个问题不是普通最短路问题

普通最短路只关心：

$$
\min_{\pi_f^{(k)}} T_{\mathrm{prop}}^{(k)}(\pi_f)  
$$

也就是单条流当前时隙的传 播时延。但在Mega-LISL场景中，一条流的路径选择会影响很多东西。

第一，它会占用链路容量：

$$
\sum_{f:e\in \pi_f^{(k)}} b_f  
\leq C_e^{(k)}  
$$

第二，它会占用卫星光终端：

$$
\sum_{f}  
\sum_{e\in\pi_f^{(k)}}  
\mathbb{I}(v\in e)  
\leq M_v  
$$

第三，它可能和其他业务流发生节点互斥。

例如两条流 $f$ 和 $g$ 不能同时经过节点 $v$，则需要满足：

$$
\chi_{f,v}^{(k)}  
+  
\chi_{g,v}^{(k)}  
\leq 1  
$$

其中：
$$
\chi_{f,v}^{(k)}

\begin{cases}
1, & v\in \pi_f^{(k)},\\
0, & v\notin \pi_f^{(k)}.
\end{cases}
$$
第四，它会影响未来是否断链、是否切换、是否集中重规划。

因此，当前最短路不一定是全局最优路由。

更关键的是，你现在要强调的核心点是：

$$
\boxed{  
\text{两对源宿如果未来会在某个互斥节点上冲突，可以让其中一条流提前切换路径，从而在时间上规避未来互斥冲突。}  
}  
$$

这就是这篇文章最有价值的地方。


# 三、时空耦合的完整定义


## 空间耦合：同一时隙内的资源竞争

在同一时隙 $k$，多条业务流同时选择路径。如果它们选择了相同链路、相同中继节点或相同光终端，就会发生资源竞争。例如链路容量约束为：

$$
\sum_{f:e\in\pi_f^{(k)}} b_f  
\leq C_e^{(k)}  
$$

节点光终端约束为：

$$
\sum_{f}  
\chi_{f,v}^{(k)}  
\leq M_v  
$$

如果某个节点是严格互斥节点，则：

$$
M_v=1  
$$

这说明同一时刻最多只能有一条业务流使用这个节点的相关资源。

空间耦合的本质是：

$$
\boxed{  
\text{一条流当前选择哪条路径，会改变其他流当前可用的路径空间。}  
}  
$$

---

## 时间耦合：当前路径影响未来代价

LEO 星座拓扑是动态变化的：

$$
\mathcal{G}^{(1)},\mathcal{G}^{(2)},\dots,\mathcal{G}^{(K)}  
$$

当前可用的链路未来可能断开：

$$
r_e^{(k)} \rightarrow 0  
$$


其中 $r_e^{(k)}$ 是链路 $e$ 在时隙 $k$ 的剩余可持续时间。路径 $\pi_f^{(k)}$ 的剩余寿命可以定义为： 
$$
R_{\min}^{(k)}(\pi_f)
=
\min_{e\in\pi_f^{(k)}} r_e^{(k)}  
$$

如果当前选择的路径很快断开，那么未来会产生被迫切换和新的 PAT 建链代价。

时间耦合的本质是：

$$
\boxed{  
\text{当前路径选择不仅决定当前时延，还决定未来的路径寿命、切换次数和建链成本。}  
}  
$$

---

## 时域互斥规避：提前切换解决未来互斥冲突

考虑两条业务流：
$$
f:(s_f,d_f)  
$$
$$
g:(s_g,d_g)  
$$
假设它们在未来时隙 $t_c$ 会同时经过互斥节点 $v$：
$$
\chi_{f,v}^{(t_c)}  
+  
\chi_{g,v}^{(t_c)}

> 1 
$$

这表示未来会发生节点互斥冲突。普通做法是等到 $t_c$ 冲突发生后再重规划，而文章方法是在$k<t_c$ 就预测到未来冲突，然后让其中一条业务流提前切换路径：

$$
\pi_f^{(k)}  
\rightarrow  
\widetilde{\pi}_f^{(k)}  
$$

使得在未来冲突窗口内：

$$
v\notin \widetilde{\pi}_f^{(t)},  
\qquad  
t\in[t_c,t_c+\Delta]  
$$

于是互斥冲突被消除：

$$
\chi_{f,v}^{(t)}  
+  
\chi_{g,v}^{(t)}  
\leq 1  
$$

这就是：

$$
\boxed{  
\text{通过提前改变一条业务流的路径占用轨迹，使两条业务流在未来冲突窗口内不再同时占用互斥节点。}  
}  
$$

注意，这不是等待，不是暂停业务，也不是让一条流延迟发送。

而是：

$$
\boxed{  
\text{路径占用轨迹的时间错开}  
}  
$$

---

# 四、核心科学问题

这篇文章可以凝练成一个核心科学问题：

$$
\boxed{  
\text{在动态 LISL 巨星座中，如何利用多智能体强化学习协调多条业务流的路径占用轨迹，使其在降低端到端时延的同时，提前规避未来节点/链路互斥冲突？}  
}  
$$

拆开来看，就是三个子问题。

---

## 子问题 1：多流并发下的资源冲突

多条业务流同时通信，路径之间会竞争链路、节点、容量和光终端。

问题是：

$$
\text{如何让不同流不要只追求自己的最短路径，而是学会相互避让？}  
$$

---

## 子问题 2：动态拓扑下的路径寿命

当前路径虽然短，但未来可能很快断开。

问题是：

$$
\text{如何让路由策略同时考虑当前时延和未来稳定性？}  
$$

---

## 子问题 3：未来互斥窗口下的主动切换

两条流当前不冲突，但未来某个窗口会冲突。

问题是：

$$
\text{如何让其中一条流在冲突发生前主动换路，从而提前释放未来互斥资源？}  
$$

这个子问题就是你这篇文章最有辨识度的创新点。

---

# 五、为什么用 MAPPO

采用Flow-Agent MAPPO，也就是：

$$
\text{每条业务流} = \text{一个智能体}  
$$

第 $f$ 个智能体对应第 $f$ 条业务流。

它的任务是：

$$
\boxed{  
\text{决定本条流在当前时隙是保持当前路径，还是提前切换到某条候选路径。}  
}  
$$

MAPPO 的结构是：

$$
\boxed{  
\text{Centralized Training + Decentralized Execution}  
}  
$$

也就是：
- 训练时，critic 可以看到全局拓扑、所有流路径、所有资源占用和未来互斥冲突；
- 执行时，每条流的 actor 根据本流观测独立输出动作。

因为互斥规避本身就是全局协同问题。对于单条流来说，提前切换可能短期看起来不划算，因为它要付出建链代价：

$$
T_{\mathrm{setup}}^{(k)}(\widetilde{\pi}_f|\pi_f)>0  
$$

但是从全局来看，它可能避免未来更严重的冲突、中断和集中重路由。

也就是：

$$
\sum_{t=k}^{k+W}  
J^{(t)}(\text{提前切换})  
<  
\sum_{t=k}^{k+W}  
J^{(t)}(\text{保持到冲突})  
$$

这个长期全局收益，正是 centralized critic 要学习的东西。

---

# 七、整体技术框架

整篇方法可以组织成下面这条主线：

$$
\boxed{  
\text{动态拓扑预测}  
\rightarrow  
\text{时空占用轨迹建模}  
\rightarrow  
\text{候选路径生成}  
\rightarrow  
\text{Flow-Agent MAPPO 决策}  
\rightarrow  
\text{资源约束执行}  
\rightarrow  
\text{时空耦合奖励反馈}  
}  
$$

具体来说：

1. 根据卫星运动获得未来窗口内的 LISL 拓扑；
    
2. 为每条业务流建立当前路径和未来占用轨迹；
    
3. 判断未来是否存在节点/链路互斥冲突；
    
4. 为每条流生成若干候选路径；
    
5. 每条流作为一个 MAPPO agent 决定保持或切换；
    
6. 环境检查链路容量、光终端和互斥约束；
    
7. 根据当前性能和未来冲突风险计算奖励；
    
8. MAPPO 更新 actor 和 centralized critic；
    
9. 最终得到一个能够主动规避未来互斥冲突的多流协同路由策略。
    

---

# 八、系统模型

## 1. 动态 LISL 网络

将 LEO 星座建模为离散时间动态图：

$$
\mathcal{G}^{(k)}

\left(  
\mathcal{V},  
\mathcal{E}^{(k)}  
\right)  
$$


其中：$\mathcal{V}$ 是卫星节点集合；$\mathcal{E}^{(k)}$ 是时隙 $k$ 下可用的星间激光链路集合。链路$e\in\mathcal{E}^{(k)}$ 具有属性：

$$
\mathbf{x}_e^{(k)}

\left[  
d_e^{(k)},  
\tau_{\mathrm{prop},e}^{(k)},  
\tau_{\mathrm{setup},e}^{(k)},  
r_e^{(k)},  
C_e^{(k)},  
u_e^{(k)}  
\right]  
$$
分别表示：
- 链路距离；
- 传播时延；
- PAT 建链时延；
- 剩余寿命；
- 链路容量；
- 当前占用状态。

---

## 2. 多源多宿业务流

业务流集合为：
$$
\mathcal{F}
=
{1,2,\dots,F}  
$$

第 $f$ 条流定义为：

$$
f=(s_f,d_f,b_f)  
$$

其中： $s_f$ 是源节点；$d_f$ 是宿节点；$b_f$ 是业务带宽需求。在时隙 $k$，第 $f$ 条流的路径为：
$$
\pi_f^{(k)}
=
(v_{f,0}^{(k)},v_{f,1}^{(k)},\dots,v_{f,H_f}^{(k)})  
$$

满足：

$$
v_{f,0}^{(k)}=s_f  
$$

$$
v_{f,H_f}^{(k)}=d_f  
$$

---

## 3. 时空占用轨迹

这是这篇文章的关键建模点。一条流不是只占用当前路径，而是在未来窗口内形成一条时空资源占用轨迹。

定义预测窗口：

$$
\mathcal{W}^{(k)}
=
{k,k+1,\dots,k+W}  
$$

第 $f$ 条流的节点占用轨迹为：

$$
\Phi_f^{(k)}
=
\left\{
(v,t)
\mid
v\in\pi_f^{(t)},
\quad
t\in\mathcal{W}^{(k)}
\right\}
$$

第 $f$ 条流的链路占用轨迹为：

$$
\Psi_f^{(k)}

\left\{
(e,t)
\mid
e\in\pi_f^{(t)},
\quad
t\in\mathcal{W}^{(k)}
\right\}  
$$

时空耦合的本质就体现在：

$$
\Phi_f^{(k)}  
\cap  
\Phi_g^{(k)}  
\neq  
\emptyset  
$$

或者：

$$
\Psi_f^{(k)}  
\cap  
\Psi_g^{(k)}  
\neq  
\emptyset  
$$

这表示两条流在未来某个时间窗口内存在资源占用重叠。

---

# 九、互斥冲突建模

对于节点互斥，定义：

$$
\mu_{f,g,v}

\begin{cases}  
1, & \text{flow } f \text{ and flow } g \text{ cannot simultaneously occupy node } v,\\  
0, & \text{otherwise}.  
\end{cases}  
$$

如果：

$$
\mu_{f,g,v}=1  
$$

则要求：

$$
\chi_{f,v}^{(t)}  
+  
\chi_{g,v}^{(t)}  
\leq 1  
$$

其中：

$$
\chi_{f,v}^{(t)}

\begin{cases}  
1, & v\in \pi_f^{(t)},\\  
0, & v\notin \pi_f^{(t)}.  
\end{cases}  
$$

未来窗口内的互斥冲突可以定义为：

$$
N_{\mathrm{mutex}}^{(k)}

\sum_{\Delta=0}^{W}  
\sum_{f=1}^{F}  
\sum_{g=f+1}^{F}  
\sum_{v\in\mathcal{V}}  
\mu_{f,g,v}  
\cdot  
\mathbb{I}  
\left(  
\chi_{f,v}^{(k+\Delta)}  
+  
\chi_{g,v}^{(k+\Delta)}

> 1  
\right)  
$$

这个公式的含义是：

从当前时隙 $k$ 往后看 $W$ 个时隙，统计所有业务流之间在互斥节点上的未来冲突次数。

这个量非常重要，因为它直接推动 MAPPO 学习提前避让。

---

# 十、提前切换规避机制

假设当前时隙为 $k$，预测到两条业务流 $f$ 和 $g$ 会在未来时隙 $t_c$ 于节点 $v$ 发生冲突：

$$
\mu_{f,g,v}=1
$$

且：

$$
\chi_{f,v}^{(t_c)}
+
\chi_{g,v}^{(t_c)}

> 1
$$

此时，如果第 $f$ 条流保持当前路径，则未来冲突不可避免。

但如果第 $f$ 条流在当前时隙主动切换：

$$
\pi_f^{(k)}
\rightarrow
\widetilde{\pi}_f^{(k)}
$$

并满足：

$$
v\notin\widetilde{\pi}_f^{(t)},
\qquad
t\in[t_c,t_c+\Delta]
$$

则可以消除未来冲突：

$$
\chi_{f,v}^{(t)}
+
\chi_{g,v}^{(t)}
\leq 1
$$

这就是本文的核心主动规避机制。

它可以写成一句论文表达：

$$
\boxed{
\text{The proposed MAPPO policy learns proactive route switching to reshape future spatio-temporal resource occupation trajectories, thereby avoiding upcoming mutual-exclusion conflicts before they occur.}
}
$$

中文就是：

$$
\boxed{
\text{所提 MAPPO 策略通过主动切换路径改变未来时空资源占用轨迹，从而在互斥冲突发生前完成规避。}
}
$$

---

# 十一、MAPPO 建模

## 1. 智能体定义

将每条业务流建模为一个智能体：

$$
\text{Agent}_f  
\leftrightarrow  
\text{Flow}_f  
$$

所以系统中共有 $F$ 个智能体。

第 $f$ 个智能体的任务是：

$$
\boxed{  
\text{为第 } f \text{ 条业务流决定当前是否保持路径或提前切换路径。}  
}  
$$

---

## 2. 状态设计

全局状态为：
$$
s^{(k)}
=
\left[  
\mathcal{G}^{(k:k+W)},  
\mathbf{\Pi}^{(k-1)},  
\mathcal{U}^{(k:k+W)},  
\mathcal{M},  
\mathcal{D},  
\mathcal{C}^{(k)}  
\right]  
$$

其中：

- $\mathcal{G}^{(k:k+W)}$：当前到未来窗口内的预测拓扑；
- $\mathbf{\Pi}^{(k-1)}$：上一时隙所有流的路径集合；
- $\mathcal{U}^{(k:k+W)}$：预测资源占用；
- $\mathcal{M}$：互斥关系矩阵；
- $\mathcal{D}={(s_f,d_f)}_{f=1}^{F}$：所有源宿对；
- $\mathcal{C}^{(k)}$：容量、光终端、链路状态。

---

## 3. 局部观测设计

第 $f$ 个 agent 的局部观测为：

$$
o_f^{(k)}
=
\left[  
s_f,  
d_f,  
\pi_f^{(k-1)},  
\mathcal{K}_f^{(k)},  
\mathbf{c}_f^{(k)},  
\mathbf{m}_f^{(k)}  
\right]  
$$

其中：
- $s_f,d_f$：该流的源宿节点；
- $\pi_f^{(k-1)}$：当前正在使用的路径；
- $\mathcal{K}_f^{(k)}$：候选路径集合；
- $\mathbf{c}_f^{(k)}$：候选路径的时延、寿命、建链代价等特征；
- $\mathbf{m}_f^{(k)}$：与该流相关的未来互斥冲突信息。


对于每条候选路径 $\pi_{f,i}^{(k)}$，构造路径特征：
$$
\mathbf{g}_{f,i}^{(k)}
=
\left[  
T_{\mathrm{prop}}^{(k)}(\pi_{f,i}),  
T_{\mathrm{setup}}^{(k)}(\pi_{f,i}|\pi_f^{(k-1)}),  
R_{\min}^{(k)}(\pi_{f,i}),  
N_{\mathrm{new}}^{(k)}(\pi_{f,i}),  
H(\pi_{f,i}),  
A_{f,i}^{(k)},  
B_{f,i}^{(k)}  
\right]  
$$

其中：
- $T_{\mathrm{prop}}$：传播时延；
- $T_{\mathrm{setup}}$：切换到该路径的 PAT 建链代价；
- $R_{\min}$：路径最小剩余寿命；
- $N_{\mathrm{new}}$：新建链路数量；
- $H$：跳数；
- $A_{f,i}^{(k)}$：该候选路径未来仍会造成的互斥冲突数量；
- $B_{f,i}^{(k)}$：该候选路径能够消除的未来互斥冲突数量。

其中，未来冲突数量可以定义为：

$$
A_{f,i}^{(k)}
=
\sum_{\Delta=0}^{W}  
\sum_{g\neq f}  
\sum_{v\in\mathcal{V}}  
\mu_{f,g,v}  
\cdot  
\mathbb{I}  
\left(  
\chi_{f,i,v}^{(k+\Delta)}  
+  
\chi_{g,v}^{(k+\Delta)}

> 1  
\right)  
$$

而规避收益可以定义为：

$$
B_{f,i}^{(k)}
=
N_{\mathrm{mutex,keep}}^{(k)}
-
N_{\mathrm{mutex,switch}}^{(k)}(\pi_{f,i})  
$$

如果：

$$
B_{f,i}^{(k)}>0  
$$

说明切换到候选路径 $i$ 可以减少未来互斥冲突。

这就是让 MAPPO 明确看见“提前切换价值”的关键特征。

---

## 4. 动作设计

动作保持简单，不要复杂化。

第 $f$ 个 agent 的动作定义为：

$$
a_f^{(k)}  
\in  
{0,1,2,\dots,K}  
$$

其中：

- $a_f^{(k)}=0$：保持当前路径；
- $a_f^{(k)}=i$：切换到第 $i$ 条候选路径 $\pi_{f,i}^{(k)}$。

路径更新为：
$$
\pi_f^{(k)}

\begin{cases}  
\pi_f^{(k-1)}, & a_f^{(k)}=0,\  
\pi_{f,i}^{(k)}, & a_f^{(k)}=i.  
\end{cases}  
$$

所有 agent 的联合动作是：

$$
\mathbf{a}^{(k)}

\left[  
a_1^{(k)},a_2^{(k)},\dots,a_F^{(k)}  
\right]  
$$

这个联合动作决定所有业务流当前的联合路径集合：

$$
\mathbf{\Pi}^{(k)}

\left[  
\pi_1^{(k)},\pi_2^{(k)},\dots,\pi_F^{(k)}  
\right]  
$$

---

# 十二、端到端时延模型

第 $f$ 条流在时隙 $k$ 的端到端时延定义为：

$$
T_f^{(k)}
=
T_{\mathrm{prop}}^{(k)}(\pi_f^{(k)})  
+  
T_{\mathrm{setup}}^{(k)}(\pi_f^{(k)}|\pi_f^{(k-1)})  
+  
T_{\mathrm{proc}}^{(k)}(\pi_f^{(k)})  
$$

传播时延为：

$$
T_{\mathrm{prop}}^{(k)}(\pi_f^{(k)})
=
\sum_{e\in\pi_f^{(k)}}  
\tau_{\mathrm{prop},e}^{(k)}  
$$

PAT 建链时延为：

$$
T_{\mathrm{setup}}^{(k)}(\pi_f^{(k)}|\pi_f^{(k-1)})
=
\begin{cases}  
\max\limits_{e\in\mathcal{L}_{\mathrm{new},f}^{(k)}}  
\tau_{\mathrm{setup},e}^{(k)},  
&  
\mathcal{L}_{\mathrm{new},f}^{(k)}\neq\emptyset,\\  
0,  
&  
\mathcal{L}_{\mathrm{new},f}^{(k)}=\emptyset.  
\end{cases}  
$$

其中：

$$
\mathcal{L}_{\mathrm{new},f}^{(k)}
=
\pi_f^{(k)}  
\setminus  
\pi_f^{(k-1)}  
$$

处理时延只与跳数有关：

$$
T_{\mathrm{proc}}^{(k)}(\pi_f^{(k)})
=
H(\pi_f^{(k)})\tau_{\mathrm{proc}}  
$$

这个时延模型里最关键的是：

$$
T_{\mathrm{setup}}^{(k)}  
$$

因为主动切换不是免费的。

MAPPO 必须自己学会：

$$
\boxed{  
\text{什么时候值得付出当前建链代价，去避免未来更大的互斥冲突代价。}  
}  
$$

---

# 十三、奖励函数设计

使用全局共享奖励：

$$
r^{(k)}

-J^{(k)}  
$$

所有 agent 共享同一个奖励。这样它们会学习协作，而不是各自只优化自己。

系统总代价定义为：

$$
J^{(k)}
=
w_1\overline{T}^{(k)}  
+  
w_2T_{\max}^{(k)}  
+  
w_3N_{\mathrm{switch}}^{(k)}  
+  
w_4N_{\mathrm{new}}^{(k)}  
+  
w_5N_{\mathrm{conflict}}^{(k)}  
+  
w_6N_{\mathrm{outage}}^{(k)}  
+  
w_7R_{\mathrm{life}}^{(k)}  
+  
w_8N_{\mathrm{mutex}}^{(k)}  
+  
w_9P_{\mathrm{sync}}^{(k)}  
$$

逐项解释如下。

---

## 1. 平均端到端时延

$$
\overline{T}^{(k)}
=
\frac{1}{F}  
\sum_{f=1}^{F}  
T_f^{(k)}  
$$

用于优化整体平均性能。

---

## 2. 峰值端到端时延

$$
T_{\max}^{(k)}
=
\max_{f\in\mathcal{F}}  
T_f^{(k)}  
$$

用于避免某一条业务流被严重牺牲。

---

## 3. 路径切换次数

$$
N_{\mathrm{switch}}^{(k)}
=
\sum_{f=1}^{F}  
\mathbb{I}  
\left(  
\pi_f^{(k)}  
\neq  
\pi_f^{(k-1)}  
\right)  
$$

用于抑制频繁切换。

---

## 4. 新建链路数量

$$
N_{\mathrm{new}}^{(k)}
=
\sum_{f=1}^{F}  
\left|  
\pi_f^{(k)}  
\setminus  
\pi_f^{(k-1)}  
\right|  
$$

用于抑制大量 PAT 建链。

---

## 5. 当前资源冲突

$$
N_{\mathrm{conflict}}^{(k)}  
$$

用于惩罚当前时隙的链路容量、光终端和资源占用冲突。

---

## 6. 业务中断数量

$$
N_{\mathrm{outage}}^{(k)}
=
\sum_{f=1}^{F}  
\mathbb{I}  
\left(  
\pi_f^{(k)}  
\text{ infeasible}  
\right)  
$$

用于惩罚无法成功建立端到端路径的业务流。

---

## 7. 路径寿命风险

$$
R_{\mathrm{life}}^{(k)}
=
\sum_{f=1}^{F}  
\frac{1}  
{  
R_{\min}^{(k)}(\pi_f^{(k)})+\epsilon  
}  
$$

用于避免选择马上断开的路径。

---

## 8. 未来互斥冲突惩罚

这是最关键的一项：

$$
N_{\mathrm{mutex}}^{(k)}
=
\sum_{\Delta=0}^{W}  
\beta^\Delta  
\sum_{f=1}^{F}  
\sum_{g=f+1}^{F}  
\sum_{v\in\mathcal{V}}  
\mu_{f,g,v}  
\cdot  
\mathbb{I}  
\left(  
\chi_{f,v}^{(k+\Delta)}  
+  
\chi_{g,v}^{(k+\Delta)}

> 1  
> \right)  
> $$

它会让 MAPPO 学会提前规避未来互斥节点冲突。

如果保持当前路径会导致未来冲突，那么这一项很大。

如果提前切换可以避开未来冲突，那么这一项变小。

因此 MAPPO 会自动比较：

$$
\text{提前切换带来的建链代价}  
$$

和：

$$
\text{不切换导致的未来互斥冲突代价}  
$$

---

## 9. 同步切换惩罚

$$
P_{\mathrm{sync}}^{(k)}
=
\left(  
N_{\mathrm{switch}}^{(k)}  
\right)^2  
$$

用于避免大量业务流在同一个时隙集中切换。

这很重要，因为并发多源多宿场景中，如果多条流同时重路由，会产生明显的建链峰值。

---

# 十四、MAPPO 网络结构

## 1. Actor

每个 actor 输入本流观测：

$$
o_f^{(k)}  
$$

输出动作概率：

$$
\pi_\theta(a_f^{(k)}|o_f^{(k)})  
$$

也就是：

$$
P(a_f^{(k)}=0),  
P(a_f^{(k)}=1),  
\dots,  
P(a_f^{(k)}=K)  
$$

其中：

- $0$ 表示保持当前路径；
- $1\sim K$ 表示切换到对应候选路径。


actor 的推荐结构是：

$$
\boxed{  
\text{候选路径特征}  
\rightarrow  
\text{MLP / Attention Encoder}  
\rightarrow  
\text{Action Head}  
\rightarrow  
\text{Softmax}  
}  
$$

如果后续想进一步增强，可以在前面加入 GNN 编码动态图，但第一版不需要把模型搞得太复杂。第一版重点应该把**时空特征、互斥冲突、提前切换收益**设计清楚。

---

## 2. Centralized Critic

critic 输入全局状态：

$$
s^{(k)}  
$$

输出状态价值：

$$
V_\phi(s^{(k)})  
$$

即：

$$
V_\phi(s^{(k)})  
\approx  
\mathbb{E}  
\left[  
\sum_{t=k}^{K}  
\gamma^{t-k}  
r^{(t)}  
\right]  
$$

critic 看到的信息包括：

- 全局拓扑；
- 所有业务流当前路径；
- 所有候选路径；
- 当前资源占用；
- 未来拓扑窗口；
- 未来互斥冲突窗口；
- 所有流的联合动作结果。
    

所以 critic 学的是：

$$
\boxed{  
\text{多条业务流联合路径决策的全局时空价值。}  
}  
$$

这正是 MAPPO 解决时空耦合问题的核心。

---

# 十五、MAPPO 训练流程

完整训练过程如下。

```text
Algorithm: MAPPO-Based Spatio-Temporal Coupled Multi-Commodity LISL Routing

Input:
    Dynamic LISL topology sequence {G^(k)}
    Flow set F = {1,2,...,F}
    Source-destination pairs {(s_f,d_f)}
    Link propagation delay, setup delay, residual lifetime, capacity
    Optical terminal constraints
    Mutual-exclusion matrix M
    Prediction window W

Initialize:
    Shared actor network π_θ
    Centralized critic network V_φ
    Experience buffer B

For each training episode do:

    Initialize dynamic topology G^(0)
    Initialize paths {π_f^(0)} for all flows

    For each time slot k = 0,1,...,K do:

        1. Obtain predicted topology window:
              G^(k:k+W)

        2. Predict spatio-temporal occupation trajectories
              for all current flows.

        3. Detect future mutual-exclusion conflicts:
              N_mutex^(k)

        4. For each flow f:
              Generate candidate path set K_f^(k)
              Extract candidate path features:
                  propagation delay
                  setup delay
                  residual lifetime
                  new-link number
                  future mutex conflict count
                  mutex avoidance benefit
              Construct local observation o_f^(k)

        5. Each actor selects action:
              a_f^(k) ~ π_θ(a_f^(k) | o_f^(k))

        6. Combine all actions:
              a^(k) = {a_1^(k),...,a_F^(k)}

        7. Environment executes joint routing decision:
              obtain Π^(k)

        8. Check constraints:
              visibility constraint
              link capacity constraint
              optical terminal constraint
              node/link mutual-exclusion constraint

        9. Compute global reward:
              r^(k) = -J^(k)

        10. Store transition:
              B ← B ∪ {s^(k), o_f^(k), a_f^(k), r^(k), s^(k+1)}

    End For

    Compute advantage estimates using centralized critic.

    Update actor using PPO clipped objective.

    Update centralized critic using value loss.

End For

Output:
    Trained MAPPO routing policy π_θ
```

---

# 十六、这篇文章的创新点

我建议最终写成三条贡献。

---

## 创新点 1：时空占用轨迹驱动的多源多宿 LISL 路由建模

本文将 LEO 巨星座中的并发多源多宿星间激光路由建模为一个时空耦合多智能体决策问题。不同于传统单时隙最短路方法，本文将每条业务流的路径表示为未来预测窗口内的时空资源占用轨迹，从而能够刻画多条业务流在链路、光终端和互斥节点上的未来冲突关系。

---

## 创新点 2：面向未来互斥冲突的主动路径切换机制

本文提出一种未来互斥冲突感知的主动切换机制。当两对源宿业务在未来冲突窗口内可能同时占用同一互斥节点或链路时，所提方法可以让其中一条业务流在冲突发生前提前切换到替代路径，从而改变其未来资源占用轨迹，在时间维度上规避节点/链路互斥冲突。

---

## 创新点 3：基于 MAPPO 的多流协同路由策略学习

本文提出 Flow-Agent MAPPO 协同路由方法，将每条业务流建模为一个智能体。每个 actor 根据本流候选路径的时延、寿命、建链代价和互斥规避收益选择保持或切换路径；centralized critic 基于全局拓扑、资源占用和未来冲突窗口学习联合路由决策的长期时空价值，从而实现低时延、低冲突、低切换峰值和高稳定性的多源多宿路由。

---

# 十七、实验设计

实验应该围绕“证明 MAPPO 学会了时空耦合规避”来设计，而不是只比较平均时延。

---

## 1. 仿真场景

设置 LEO 巨星座动态拓扑，考虑：

- 不同星间激光链路最大距离；
- 不同业务流数量 $F$；
- 不同源宿分布；
- 不同链路容量；
- 不同光终端数量；
- 不同节点互斥强度；
- 不同预测窗口长度 $W$。
    

---

## 2. 对比方法

建议设置以下 baseline：

1. **Independent Shortest-Delay Routing**
    

每条流独立选择当前时延最短路径，不考虑其他流未来冲突。

2. **Conflict-Aware Greedy Routing**
    

每个时隙只避开当前冲突，但不考虑未来互斥窗口。

3. **Maintain-until-Failure Rerouting**
    

路径可用就保持，直到断链或冲突后才重规划。

4. **Single-Agent PPO Routing**
    

用单智能体处理联合动作，作为强化学习但非多智能体协同的对比。

5. **Proposed MAPPO**
    

你的方法。

这几个 baseline 的逻辑是递进的，能够证明：

$$
\text{MAPPO 的提升来自多流协同和时空预测，而不是简单强化学习。}  
$$

---

## 3. 评价指标

必须包括以下指标。

### 平均端到端时延

$$
\overline{T}
=
\frac{1}{KF}  
\sum_{k=1}^{K}  
\sum_{f=1}^{F}  
T_f^{(k)}  
$$

### 峰值端到端时延

$$
T_{\mathrm{peak}}
=
\max_{k,f}  
T_f^{(k)}  
$$

### 切换次数

$$
N_{\mathrm{switch,total}}
=
\sum_{k=1}^{K}  
N_{\mathrm{switch}}^{(k)}  
$$

### 新建链路数量

$$
N_{\mathrm{new,total}}
=
\sum_{k=1}^{K}  
N_{\mathrm{new}}^{(k)}  
$$

### 互斥冲突次数

$$
N_{\mathrm{mutex,total}}
=
\sum_{k=1}^{K}  
N_{\mathrm{mutex}}^{(k)}  
$$

这个指标非常重要，必须单独画图。

### 业务中断率

$$
P_{\mathrm{outage}}
=
\frac{  
\sum_{k=1}^{K}N_{\mathrm{outage}}^{(k)}  
}{  
KF  
}  
$$

### 同步切换峰值

$$
P_{\mathrm{sync,peak}}
=
\max_k  
N_{\mathrm{switch}}^{(k)}  
$$

这个指标用于证明 MAPPO 能减少集中重路由。

---

## 4. 关键实验图

我建议至少设计六类图。

---

### 图 1：不同业务流数量下的平均时延

横轴：

$$
F  
$$

纵轴：

$$
\overline{T}  
$$

目的：证明业务流越多，资源竞争越强，MAPPO 的优势越明显。

---

### 图 2：不同业务流数量下的互斥冲突次数

横轴：

$$
F  
$$

纵轴：

$$
N_{\mathrm{mutex,total}}  
$$

目的：直接证明 MAPPO 能减少未来互斥冲突。

这是支撑你核心创新的关键图。

---

### 图 3：不同预测窗口长度下的性能

横轴：

$$
W  
$$

纵轴可以画：

- 平均时延；
- 互斥冲突次数；
- 切换次数。
    

目的：证明预测窗口过短看不到未来冲突，窗口适中时效果最好。

---

### 图 4：主动切换案例图

这张图最重要，应该做成 case study。

展示两条业务流：

$$
f:(s_f,d_f)  
$$

$$
g:(s_g,d_g)  
$$

它们如果保持原路径，会在未来时隙 $t_c$ 同时经过互斥节点 $v$。

然后展示 MAPPO 在 $k<t_c$ 时刻让其中一条流提前切换，最终避免冲突。

这张图要表达：

$$
\boxed{  
\text{MAPPO 不是等冲突发生再处理，而是在冲突窗口到来之前提前改变路径占用轨迹。}  
}  
$$

---

### 图 5：同步切换峰值对比

横轴：

$$
k  
$$

纵轴：

$$
N_{\mathrm{switch}}^{(k)}  
$$

目的：证明 MAPPO 能降低集中重规划。

---

### 图 6：消融实验

比较：

1. MAPPO without mutex penalty；
    
2. MAPPO without residual lifetime；
    
3. MAPPO without sync switching penalty；
    
4. Full MAPPO。
    

目的：证明每一项设计都有作用，尤其是：

$$
N_{\mathrm{mutex}}^{(k)}  
$$

这一项必须通过消融实验体现价值。

---

# 十八、论文结构规划

这篇文章可以按下面结构写。

---

## 1. Introduction

重点写问题链：

1. LEO 巨星座星间激光通信需要动态路由；
    
2. 多源多宿并发业务导致链路、光终端和节点资源竞争；
    
3. 动态拓扑使当前路由决策影响未来路径寿命和切换代价；
    
4. 更重要的是，不同业务流可能在未来窗口内发生节点/链路互斥冲突；
    
5. 等冲突发生后再重规划会导致集中切换、高建链代价和业务中断；
    
6. 因此需要一种能学习未来时空资源占用轨迹的多流协同路由方法；
    
7. 本文提出 Flow-Agent MAPPO。
    

---

## 2. System Model

包括：

- 动态 LISL 网络模型；
- 多源多宿业务流模型；
- 链路传播时延；
- PAT 建链时延；
- 光终端和容量约束；
- 节点/链路互斥约束；
- 时空占用轨迹定义。
    

---

## 3. Problem Formulation

定义优化目标：

$$
\min_{{\mathbf{\Pi}^{(k)}}_{k=1}^{K}}  
\sum_{k=1}^{K}  
J^{(k)}  
$$

并说明这是一个动态、多流、资源约束、长期累计决策问题，很难用传统单时隙优化方法直接解决。

---

## 4. MAPPO-Based Spatio-Temporal Coupled Routing

包括：

- flow-agent 建模；
- 状态设计；
- 局部观测；
- 动作设计；
- 奖励函数；
- actor 网络；
- centralized critic；
- MAPPO 训练流程。
    

---

## 5. Experiments and Results

包括：

- 仿真设置；
- baseline；
- 评价指标；
- 平均时延对比；
- 互斥冲突对比；
- 主动切换案例；
- 消融实验；
- 复杂度和泛化性分析。
    

---

## 6. Conclusion

总结：

- 本文解决了动态 LISL 多源多宿时空耦合路由问题；
- 提出了时空占用轨迹建模；
- 通过 MAPPO 学习主动切换和互斥规避；
- 实验验证降低了平均时延、峰值时延、互斥冲突、切换峰值和中断率。
    

---

# 十九、最终核心摘要版本

你后面和老师讨论时，可以先用这段话概括：

本文研究 LEO 巨星座星间激光通信中的并发多源多宿路由问题。与传统单源单宿最短路不同，多条业务流在同一动态拓扑中同时通信，会竞争有限的链路容量、光终端和中继节点资源，并且当前路径选择会影响未来路径寿命、切换代价和资源占用关系。尤其在存在节点或链路互斥的源宿对之间，两条业务流可能在未来某个时间窗口内同时占用同一互斥资源，导致冲突或中断。为此，本文将每条业务流建模为一个智能体，提出基于 MAPPO 的时空耦合多源多宿协同路由方法。该方法利用 centralized critic 学习全局拓扑演化、资源占用和未来互斥冲突窗口，并通过各业务流 actor 决定保持当前路径或提前切换到候选路径。通过主动改变业务流的未来时空资源占用轨迹，所提方法可以在互斥冲突发生前完成规避，从而降低平均端到端时延、峰值时延、切换次数、互斥冲突和业务中断率。

---

# 二十、最终一句话定位

最后，这篇文章最核心的定位就是：

$$
\boxed{  
\text{不是用 MAPPO 替代最短路，而是用 MAPPO 学习多条业务流在动态星间激光网络中的时空占用轨迹协调，使部分业务流能够在未来互斥冲突发生前主动切换路径，从而实现并发多源多宿通信的低冲突、低时延和高稳定路由。}  
}  
$$
