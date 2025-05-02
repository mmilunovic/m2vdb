import React, { useState } from 'react';

interface VectorControlsProps {
  onAddVector: (text: string, metadata: { category: string; source: string }) => void;
  onSearch: (query: string) => void;
}

const VectorControls: React.FC<VectorControlsProps> = ({
  onAddVector,
  onSearch,
}) => {
  const [newDocument, setNewDocument] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [metadata, setMetadata] = useState({
    category: '',
    source: ''
  });
  const [showMetadata, setShowMetadata] = useState(false);

  const handleAddDocument = async () => {
    if (!newDocument.trim()) return;
    setIsLoading(true);
    try {
      await onAddVector(newDocument, metadata);
      setNewDocument('');
      setMetadata({ category: '', source: '' });
      setShowMetadata(false);
    } catch (error) {
      console.error('Error adding document:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    try {
      await onSearch(searchQuery);
    } catch (error) {
      console.error('Error searching:', error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent, type: 'add' | 'search') => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (type === 'add') {
        handleAddDocument();
      } else {
        handleSearch();
      }
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: '24px',
      left: '24px',
      width: '360px',
      backgroundColor: '#ffffff',
      borderRadius: '16px',
      boxShadow: '0 4px 12px rgba(0, 0, 0, 0.08)',
      zIndex: 1000,
      padding: '20px',
      display: 'flex',
      flexDirection: 'column',
      gap: '16px'
    }}>
      {/* Add Document Section */}
      <div>
        <div style={{
          fontSize: '13px',
          color: '#787774',
          marginBottom: '8px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          Add Content
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <textarea
            value={newDocument}
            onChange={(e) => setNewDocument(e.target.value)}
            onKeyPress={(e) => handleKeyPress(e, 'add')}
            placeholder="Type text to embed in the database..."
            style={{
              width: '100%',
              height: '80px',
              padding: '12px',
              fontSize: '14px',
              lineHeight: '1.5',
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              backgroundColor: '#fff',
              resize: 'none',
              fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
              transition: 'all 0.2s ease',
              outline: 'none',
              color: '#37352f',
            }}
            onFocus={e => {
              e.target.style.borderColor = '#b2bac2';
              e.target.style.boxShadow = '0 0 0 2px rgba(0,0,0,0.05)';
            }}
            onBlur={e => {
              e.target.style.borderColor = '#e0e0e0';
              e.target.style.boxShadow = '0 1px 2px rgba(0,0,0,0.02)';
            }}
          />
          <div style={{
            display: 'flex',
            gap: '8px',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <button
              onClick={() => setShowMetadata(!showMetadata)}
              style={{
                background: 'none',
                border: 'none',
                color: '#787774',
                fontSize: '12px',
                cursor: 'pointer',
                padding: '4px 8px',
                borderRadius: '4px',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                transition: 'all 0.2s ease',
              }}
              onMouseOver={e => e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.04)'}
              onMouseOut={e => e.currentTarget.style.backgroundColor = 'transparent'}
            >
              {showMetadata ? '▼' : '▶'} Add metadata
            </button>
            <button
              onClick={handleAddDocument}
              disabled={isLoading || !newDocument.trim()}
              style={{
                height: '32px',
                padding: '0 16px',
                backgroundColor: '#37352f',
                color: '#fff',
                border: 'none',
                borderRadius: '8px',
                cursor: isLoading || !newDocument.trim() ? 'not-allowed' : 'pointer',
                fontSize: '14px',
                fontWeight: 500,
                opacity: isLoading || !newDocument.trim() ? 0.5 : 1,
                transition: 'all 0.2s ease',
                whiteSpace: 'nowrap',
              }}
              onMouseOver={e => {
                if (!isLoading && newDocument.trim()) {
                  e.currentTarget.style.backgroundColor = '#2a2a2a';
                }
              }}
              onMouseOut={e => {
                if (!isLoading && newDocument.trim()) {
                  e.currentTarget.style.backgroundColor = '#37352f';
                }
              }}
            >
              {isLoading ? 'Adding...' : 'Add Content'}
            </button>
          </div>
          {showMetadata && (
            <div style={{
              padding: '12px',
              backgroundColor: '#f7f6f3',
              borderRadius: '8px',
              display: 'flex',
              flexDirection: 'column',
              gap: '8px',
              border: '1px solid rgba(0, 0, 0, 0.05)',
            }}>
              <input
                type="text"
                value={metadata.category}
                onChange={e => setMetadata(prev => ({ ...prev, category: e.target.value }))}
                placeholder="Category (e.g., note, article, code)"
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #e0e0e0',
                  borderRadius: '6px',
                  fontSize: '13px',
                  backgroundColor: '#fff',
                  color: '#37352f',
                  height: '32px',
                }}
              />
              <input
                type="text"
                value={metadata.source}
                onChange={e => setMetadata(prev => ({ ...prev, source: e.target.value }))}
                placeholder="Source (e.g., web, local, api)"
                style={{
                  width: '100%',
                  padding: '8px 12px',
                  border: '1px solid #e0e0e0',
                  borderRadius: '6px',
                  fontSize: '13px',
                  backgroundColor: '#fff',
                  color: '#37352f',
                  height: '32px',
                }}
              />
            </div>
          )}
        </div>
      </div>

      {/* Search Section */}
      <div>
        <div style={{
          fontSize: '13px',
          color: '#787774',
          marginBottom: '8px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px'
        }}>
          Search
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '12px'
        }}>
          <textarea
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => handleKeyPress(e, 'search')}
            placeholder="Search for similar vectors..."
            style={{
              width: '100%',
              height: '80px',
              padding: '12px',
              fontSize: '14px',
              lineHeight: '1.5',
              border: '1px solid #e0e0e0',
              borderRadius: '8px',
              backgroundColor: '#fff',
              resize: 'none',
              fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              boxShadow: '0 1px 2px rgba(0,0,0,0.02)',
              transition: 'all 0.2s ease',
              outline: 'none',
              color: '#37352f',
            }}
            onFocus={e => {
              e.target.style.borderColor = '#b2bac2';
              e.target.style.boxShadow = '0 0 0 2px rgba(0,0,0,0.05)';
            }}
            onBlur={e => {
              e.target.style.borderColor = '#e0e0e0';
              e.target.style.boxShadow = '0 1px 2px rgba(0,0,0,0.02)';
            }}
          />
          <button
            onClick={handleSearch}
            disabled={!searchQuery.trim()}
            style={{
              height: '32px',
              width: '100%',
              backgroundColor: '#37352f',
              color: '#fff',
              border: 'none',
              borderRadius: '8px',
              cursor: !searchQuery.trim() ? 'not-allowed' : 'pointer',
              fontSize: '14px',
              fontWeight: 500,
              opacity: !searchQuery.trim() ? 0.5 : 1,
              transition: 'all 0.2s ease',
            }}
            onMouseOver={e => {
              if (searchQuery.trim()) {
                e.currentTarget.style.backgroundColor = '#2a2a2a';
              }
            }}
            onMouseOut={e => {
              if (searchQuery.trim()) {
                e.currentTarget.style.backgroundColor = '#37352f';
              }
            }}
          >
            Search
          </button>
        </div>
      </div>
    </div>
  );
};

export default VectorControls; 