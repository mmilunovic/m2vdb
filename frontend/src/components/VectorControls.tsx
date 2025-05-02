import React, { useState } from 'react';
import { Input, Button, Space, Card } from 'antd';

const { TextArea } = Input;

interface VectorControlsProps {
  onAddDocument: (text: string, metadata: any) => void;
  onSearch: (query: string) => void;
}

const VectorControls: React.FC<VectorControlsProps> = ({
  onAddDocument,
  onSearch,
}) => {
  const [newDocument, setNewDocument] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  const handleAddDocument = () => {
    if (newDocument.trim()) {
      onAddDocument(newDocument, { timestamp: new Date().toISOString() });
      setNewDocument('');
    }
  };

  const handleSearch = () => {
    if (searchQuery.trim()) {
      onSearch(searchQuery);
    }
  };

  return (
    <Space direction="vertical" style={{ width: '100%' }} size="large">
      <Card title="Add New Document">
        <Space direction="vertical" style={{ width: '100%' }}>
          <TextArea
            value={newDocument}
            onChange={(e) => setNewDocument(e.target.value)}
            placeholder="Enter document text..."
            rows={4}
          />
          <Button type="primary" onClick={handleAddDocument}>
            Add Document
          </Button>
        </Space>
      </Card>

      <Card title="Search Documents">
        <Space direction="vertical" style={{ width: '100%' }}>
          <Input
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Enter search query..."
            onPressEnter={handleSearch}
          />
          <Button type="primary" onClick={handleSearch}>
            Search
          </Button>
        </Space>
      </Card>
    </Space>
  );
};

export default VectorControls; 