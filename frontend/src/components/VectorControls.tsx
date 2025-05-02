import React, { useState } from 'react';

interface VectorControlsProps {
  onAddVector: (text: string, metadata: { category: string; source: string }) => void;
}

const VectorControls: React.FC<VectorControlsProps> = ({
  onAddVector,
}) => {
  const [newDocument, setNewDocument] = useState('');
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

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleAddDocument();
    }
  };

  return (
    <div style={{
      position: 'fixed',
      top: 0,
      left: 0,
      right: 0,
      padding: '16px',
      backgroundColor: '#ffffff',
      borderBottom: '1px solid rgba(0, 0, 0, 0.1)',
      zIndex: 1000
    }}>
      <div style={{
        maxWidth: '800px',
        margin: '0 auto',
        position: 'relative'
      }}>
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          gap: '12px',
        }}>
          <textarea
            value={newDocument}
            onChange={(e) => setNewDocument(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type something to add to the vector database..."
            style={{
              flex: 1,
              minHeight: '32px',
              maxHeight: '80px',
              padding: '6px 10px',
              fontSize: '15px',
              lineHeight: '1.5',
              border: '1px solid #e0e0e0',
              borderRadius: '4px',
              backgroundColor: '#fff',
              resize: 'vertical',
              fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              boxShadow: '0 1px 2px rgba(0,0,0,0.04)',
              transition: 'border-color 0.2s, box-shadow 0.2s',
              outline: 'none',
              color: '#37352f',
              margin: 0,
              display: 'block',
            }}
            onFocus={e => {
              e.target.style.borderColor = '#b2bac2';
              e.target.style.boxShadow = '0 0 0 2px #e7e9ea';
            }}
            onBlur={e => {
              e.target.style.borderColor = '#e0e0e0';
              e.target.style.boxShadow = '0 1px 2px rgba(0,0,0,0.04)';
            }}
          />
          <button
            onClick={handleAddDocument}
            disabled={isLoading || !newDocument.trim()}
            style={{
              height: '32px',
              minWidth: '56px',
              maxWidth: '80px',
              padding: '0 12px',
              backgroundColor: '#37352f',
              color: '#fff',
              border: 'none',
              borderRadius: '4px',
              cursor: isLoading || !newDocument.trim() ? 'not-allowed' : 'pointer',
              fontSize: '15px',
              fontWeight: 500,
              opacity: isLoading || !newDocument.trim() ? 0.5 : 1,
              transition: 'opacity 0.2s',
              margin: 0,
              whiteSpace: 'nowrap',
              display: 'block',
            }}
          >
            {isLoading ? 'Adding...' : 'Add'}
          </button>
        </div>
        <div style={{
          display: 'flex',
          flexDirection: 'row',
          alignItems: 'center',
          gap: '16px',
          marginTop: '6px',
          marginLeft: '2px',
        }}>
          <button
            onClick={() => setShowMetadata(!showMetadata)}
            style={{
              background: 'none',
              border: 'none',
              color: '#787774',
              fontSize: '13px',
              cursor: 'pointer',
              padding: '2px 4px',
              borderRadius: '3px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              margin: 0,
            }}
            onMouseOver={e => e.currentTarget.style.backgroundColor = 'rgba(0,0,0,0.04)'}
            onMouseOut={e => e.currentTarget.style.backgroundColor = 'transparent'}
          >
            {showMetadata ? '▼' : '▶'} Add metadata
          </button>
          <span style={{ fontSize: '13px', color: '#787774' }}>
            Press <b>Enter</b> to add
          </span>
        </div>
        {showMetadata && (
          <div style={{
            marginTop: '12px',
            padding: '12px',
            backgroundColor: '#f7f6f3',
            borderRadius: '6px',
            display: 'flex',
            gap: '12px',
          }}>
            <input
              type="text"
              value={metadata.category}
              onChange={e => setMetadata(prev => ({ ...prev, category: e.target.value }))}
              placeholder="Category (e.g., note, article, code)"
              style={{
                flex: 1,
                padding: '8px 12px',
                border: '1px solid #e0e0e0',
                borderRadius: '4px',
                fontSize: '14px',
                backgroundColor: '#fff',
                color: '#37352f',
                height: '36px',
              }}
            />
            <input
              type="text"
              value={metadata.source}
              onChange={e => setMetadata(prev => ({ ...prev, source: e.target.value }))}
              placeholder="Source (e.g., web, local, api)"
              style={{
                flex: 1,
                padding: '8px 12px',
                border: '1px solid #e0e0e0',
                borderRadius: '4px',
                fontSize: '14px',
                backgroundColor: '#fff',
                color: '#37352f',
                height: '36px',
              }}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default VectorControls; 