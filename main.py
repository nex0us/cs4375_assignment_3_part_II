import chardet
import re
import html
import random
from collections import Counter
import os

class K_means:
    def __init__(self, filename, K, max_iter=1000, random_seed=0):
        self.K = K
        self.max_iter = max_iter
        random.seed(random_seed)

        self.tweets = self.load_file(filename)
        self.clean_tweets()
        self.write_file(f'{os.path.splitext(filename)[0]}_cleaned.txt', self.tweets)

        centroids, clusters, sse, cluster_counts = self.cluster()
        self.print_results(K, sse, cluster_counts)

    def load_file(self, filename):
        with open(filename, 'rb') as file:
            raw_data = file.read(10000)
            result = chardet.detect(raw_data)
            encoding = result['encoding']

        with open(filename, 'r', encoding=encoding) as file:
            lines = file.readlines()
            return lines
    
    def write_file(self, filename, content):
        with open(filename, 'w') as file:
            file.writelines([line + '\n' for line in content])

    def clean_tweets(self):
        for i in range(len(self.tweets)):
            self.tweets[i] = html.unescape(self.tweets[i]) # Decode html
            self.tweets[i] = re.sub(r'\[dot\]', '.', self.tweets[i])
            self.tweets[i] = re.sub(r'\[at\]', '@', self.tweets[i])
            self.tweets[i] = re.sub(r'^\d+\|', '', self.tweets[i]) # Remove id
            self.tweets[i] = re.sub(r'^[a-zA-Z]{3}\s+[a-zA-Z]{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s[+\-]\d{4}\s\d{4}\|', '', self.tweets[i]) # Remove timestamp
            self.tweets[i] = re.sub(r'^\s*RT\s*[:\-\s]*', '', self.tweets[i]) # Remove RT
            self.tweets[i] = re.sub(r'@\w+:?', '', self.tweets[i]) # Remove words starting w/ @
            self.tweets[i] = re.sub(r'#', '', self.tweets[i]) # Remove # symbols
            self.tweets[i] = re.sub(r'http[s]?://[^\s]+', '', self.tweets[i]) # Remove URLs
            self.tweets[i] = re.sub(r'\S+\.(co|com)\b', '', self.tweets[i])  # Remove strings ending in .co or .com (URLs)
            self.tweets[i] = self.tweets[i].lower() # Convert to lowercase
            self.tweets[i] = re.sub(r'\s+', ' ', self.tweets[i]).strip() # Remove extra space

    def jaccard_dist(self, tweet1, tweet2):
        # convert to sets
        set1 = set(tweet1.split())
        set2 = set(tweet2.split())
        
        # calculate parts of the formula
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        if len(union) == 0: # Handle both empty sets
            return 0.0
        return 1 - len(intersection) / len(union)

    def sum_squared_error(self, centroids, clusters):
        sse = 0
        for i in range(self.K):
            cluster_points = [self.tweets[j] for j in range(len(self.tweets)) if clusters[j] == i]
            
            if len(cluster_points) == 0:
                continue
            
            centroid = centroids[i]
            
            for point in cluster_points:
                sse += self.jaccard_dist(point, centroid) ** 2
        
        return sse

    def init_centroids(self):
        random_tweets = random.sample(range(len(self.tweets)), self.K)
        return [self.tweets[i] for i in random_tweets]
    
    def assign_clusters(self, centroids):
        clusters = []
        for tweet in self.tweets:
            distances = [self.jaccard_dist(tweet, centroid) for centroid in centroids]
            clusters.append(distances.index(min(distances)))

        return clusters
    
    def compute_centroid(self, cluster_points):
        centroid_distances = []

        for tweet in cluster_points:
            distance_sum = sum(self.jaccard_dist(tweet, other) for other in cluster_points)
            centroid_distances.append((tweet, distance_sum))

        new_centroid = min(centroid_distances, key=lambda x: x[1])[0]
        return new_centroid
    
    def update_centroids(self, clusters, centroids):
        new_centroids = []
        for i in range(self.K):
            cluster_points = [self.tweets[j] for j in range(len(self.tweets)) if clusters[j] == i]
            if cluster_points:

                new_centroid = self.compute_centroid(cluster_points)
                new_centroids.append(new_centroid)
            else:

                new_centroids.append(centroids[i])

        return new_centroids

    # def convergence(self, centroids, prev_centroids):
    #     return centroids == prev_centroids

    def convergence(self, centroids, prev_centroids, threshold=0.0001):
        if len(centroids) != len(prev_centroids):
            return False
        
        for curr, prev in zip(centroids, prev_centroids):
            if self.jaccard_dist(curr, prev) > threshold:
                return False
        
        return True
    
    def cluster(self):
        centroids = self.init_centroids()
        clusters = [None] * len(self.tweets)

        for i in range(self.max_iter):
            clusters = self.assign_clusters(centroids)
            new_centroids = self.update_centroids(clusters, centroids)

            if self.convergence(new_centroids, centroids):
                # print(f'Converged: [{i+1}]')
                break

            centroids = new_centroids

            # See SSE for every iteration
            # sse = self.sum_squared_error(centroids, clusters)
            # print(f'SSE: {sse}')

        sse = self.sum_squared_error(centroids, clusters)

        cluster_counts = self.cluster_counts(clusters)
    
        return centroids, clusters, sse, cluster_counts
    
    def cluster_counts(self, clusters):
        return Counter(clusters)

    def print_cluster_sizes(self, buffer, cluster_counts):
        for cluster in sorted(cluster_counts.keys()):
            print(f'{buffer if cluster != 0 else ""}{cluster + 1}: {cluster_counts[cluster]} tweets')

    def print_results(self, K, sse, cluster_counts):
        divider = '  ||  '
        print(K, end=divider)
        print(sse, end=divider)
        buffer = " " * (len(str(K)) + len(divider)*2 + len(str(sse)))
        self.print_cluster_sizes(buffer, cluster_counts)
        print()

if __name__ == "__main__":
    filename = 'usnewshealth.txt'
    random_seed = 10
    max_iter = 100
    k_means = K_means(filename, K=10, random_seed=random_seed, max_iter=max_iter)
    k_means = K_means(filename, K=20, random_seed=random_seed, max_iter=max_iter)
    k_means = K_means(filename, K=50, random_seed=random_seed, max_iter=max_iter)
    k_means = K_means(filename, K=100, random_seed=random_seed, max_iter=max_iter)
    k_means = K_means(filename, K=150, random_seed=random_seed, max_iter=max_iter)