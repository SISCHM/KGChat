    @staticmethod
    def split_label(text, max_line_length=10):
        words = text.split(' ')
        lines = []
        current_line = words[0]
        for word in words[1:]:
            if len(current_line) + len(word) + 1 <= max_line_length:
                current_line += ' ' + word
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)
        return '\n'.join(lines)


    def visualize_graph(self):
        g = nx.DiGraph()
        for i, node in self.nodes.iterrows():
            label = self.split_label(node['node_attr'])
            g.add_node(i, label=label, average_cost=node['average_cost'], average_invoiced_price=node['average_invoiced_price'])

        for i, row in self.edges.iterrows():
            g.add_edge(row['src'], row['dst'], frequency=row['freq'], avg_time=row['avg_time'])
        
        '''''
        pos = nx.spring_layout(g, iterations=200, k=0.1)  # Standard 
        pos = nx.kamada_kawai_layout(g)  # Good one
        pos = nx.planar_layout(g) # Edges dont cross (main graph is not planar)
        pos = nx.circular_layout(g)
        pos = nx.shell_layout(g)
        pos = nx.spectral_layout(g)
        pos = nx.random_layout(g)
        pos = nx.spiral_layout(g)

         '''''
        pos = nx.spring_layout(g, iterations=200, k=5)

        # adjust node size based on the label 
        node_labels = nx.get_node_attributes(g, 'label')
        node_sizes = {node: len(label) * 200 for node, label in node_labels.items()}  # Play with the factor 200

        # adjust edge positions 
        ax = plt.gca()
        for edge in g.edges(data=True):
            src, dst = edge[0], edge[1]
            arrowprops = dict(arrowstyle='-|>', color='grey')
            ax.annotate('', xy=pos[dst], xytext=pos[src], arrowprops=arrowprops)


        nx.draw_networkx_nodes(g, pos, node_size=[node_sizes[node] for node in g.nodes()], node_color="lightblue", edgecolors='k', linewidths=1)
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=10, font_color='black', font_weight='bold')

        edge_labels = {(src, dst): f'Freq: {data["frequency"]}\nAvg Time: {data["avg_time"]}' for src, dst, data in g.edges(data=True)}
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, font_color='black')

        plt.axis('off')  
        plt.show()    