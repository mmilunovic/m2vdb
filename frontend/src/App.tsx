import React, { useState, useEffect } from 'react';
import VectorVisualization from './components/VectorVisualization';
import VectorControls from './components/VectorControls';
import axios from 'axios';
import './App.css';

interface VectorData {
  id: number;
  text: string;
  vector: number[];
  metadata: {
    source: string;
    category: string;
    [key: string]: any;
  };
}

interface SearchResult {
  id: number;
  text: string;
  similarity_score: number;
  metadata: {
    source: string;
    category: string;
  };
}

interface SearchResponse {
  query: string;
  results: SearchResult[];
  stats: {
    total_vectors_searched: number;
    k: number;
    metric: string;
  };
  notification: string | null;
}

function App() {
  const [vectors, setVectors] = useState<VectorData[]>([]);
  const [selectedVector, setSelectedVector] = useState<VectorData | null>(null);
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [searchQuery, setSearchQuery] = useState<string>('');

  // Fetch all vectors on mount
  useEffect(() => {
    const fetchVectors = async () => {
      try {
        const response = await axios.get('http://localhost:8000/vectors');
        const vectorsData = response.data.vectors || response.data;
        setVectors(Array.isArray(vectorsData) ? vectorsData : []);
      } catch (error) {
        console.error('Error fetching vectors:', error);
      }
    };
    fetchVectors();
  }, []);

  // Add new vector and auto-select it
  const handleAddVector = async (text: string, metadata: { category: string; source: string }) => {
    try {
      await axios.post('http://localhost:8000/add_text', {
        text,
        metadata: {
          ...metadata,
          timestamp: new Date().toISOString()
        }
      });
      // Refetch all vectors (or just append if backend returns the new one)
      const vectorsResponse = await axios.get('http://localhost:8000/vectors');
      const vectorsData = vectorsResponse.data.vectors || vectorsResponse.data;
      setVectors(Array.isArray(vectorsData) ? vectorsData : []);
      // Auto-select the newest vector (assume it's last)
      if (Array.isArray(vectorsData) && vectorsData.length > 0) {
        setSelectedVector(vectorsData[vectorsData.length - 1]);
      }
    } catch (error) {
      console.error('Error adding document:', error);
    }
  };

  // Handle search
  const handleSearch = async (query: string) => {
    try {
      setSearchQuery(query);
      const response = await axios.post<SearchResponse>('http://localhost:8000/search_text', {
        query: query,
        k: 5
      });
      setSearchResults(response.data.results);
      // Clear selected vector when searching
      setSelectedVector(null);
    } catch (error) {
      console.error('Error searching:', error);
    }
  };

  return (
    <div className="App">
      <VectorControls onAddVector={handleAddVector} onSearch={handleSearch} />
      <div style={{ marginTop: '92px' }}>
        <VectorVisualization
          vectors={vectors}
          setVectors={setVectors}
          selectedVector={selectedVector}
          setSelectedVector={setSelectedVector}
          searchResults={searchResults}
          searchQuery={searchQuery}
        />
      </div>
    </div>
  );
}

export default App; 