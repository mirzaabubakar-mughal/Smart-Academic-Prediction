import React, { useState, useEffect, useCallback } from 'react';
import { 
  FaDatabase, 
  FaBrain, 
  FaChartLine, 
  FaChartBar, 
  FaHistory, 
  FaFileAlt,
  FaCircle
} from 'react-icons/fa';

import { ENDPOINTS, apiCall } from './utils/api';
import './App.css';

// Import tab components
import DatasetTab from './components/Dataset';
import TrainingTab from './components/Training';
import PredictionTab from './components/Prediction';
import VisualizationTab from './components/Visualization';
import HistoryTab from './components/History';
import ReportTab from './components/Report';

function App() {
  // ============ STATE ============
  const [activeTab, setActiveTab] = useState('dataset');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState({ type: '', text: '' });
  const [metrics, setMetrics] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const [modelsTrained, setModelsTrained] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [trainingStatus, setTrainingStatus] = useState('idle');
  const [smotePreview, setSmotePreview] = useState(null);
  const [confusionMatrices, setConfusionMatrices] = useState({});
  const [rocAucImages, setRocAucImages] = useState({});
  const [graphImages, setGraphImages] = useState({});
  const [showSmotePreview, setShowSmotePreview] = useState(false);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [modelComparison, setModelComparison] = useState([]);
  const [dbConnected, setDbConnected] = useState(false);

  // ============ EFFECTS ============
  // Check database connection on mount
  useEffect(() => {
    const checkDbConnection = async () => {
      try {
        const data = await apiCall(ENDPOINTS.DB_STATUS);
        setDbConnected(data.connected || false);
      } catch {
        setDbConnected(false);
      }
    };
    checkDbConnection();
  }, []);

  // ============ HELPERS ============
  const showMessage = useCallback((type, text) => {
    setMessage({ type, text });
    setTimeout(() => setMessage({ type: '', text: '' }), 4000);
  }, []);

  const forceRefresh = useCallback(() => {
    setRefreshTrigger(prev => prev + 1);
  }, []);

  const updateTrainingStatus = useCallback((status, success = false) => {
    setTrainingStatus(status);
    if (success) setModelsTrained(true);
    forceRefresh();
  }, [forceRefresh]);

  // ============ DATASET FUNCTIONS ============
  const previewSmoteBalanced = async () => {
    setLoading(true);
    try {
      const result = await apiCall(ENDPOINTS.DATASET_SMOTE);
      setSmotePreview(result);
      setShowSmotePreview(true);
      showMessage('success', 'SMOTE balanced dataset preview loaded');
    } catch {
      showMessage('error', 'Error loading SMOTE preview');
    } finally {
      setLoading(false);
    }
  };

  // ============ VISUALIZATION FUNCTIONS ============
  const loadConfusionMatrix = async (modelName) => {
    try {
      const url = `${API_URL}/visualization/confusion_matrix/${modelName}`;
      const result = await apiCall(url);
      setConfusionMatrices(prev => ({ ...prev, [modelName]: result.image }));
    } catch (err) {
      console.error('Error loading confusion matrix:', err);
    }
  };

  const loadAllConfusionMatrices = async () => {
    if (!metrics) return;
    const models = Object.keys(metrics);
    for (const model of models) {
      await loadConfusionMatrix(model);
    }
    showMessage('success', `Loaded ${models.length} confusion matrices`);
  };

  const loadAllRocAuc = async () => {
    if (!metrics) return;
    setLoading(true);
    const allModels = Object.keys(metrics);
    for (const model of allModels) {
      try {
        const url = `${API_URL}/visualization/roc_auc/${model}`;
        const result = await apiCall(url);
        setRocAucImages(prev => ({ ...prev, [model]: result.image }));
      } catch (err) {
        console.error(`Error loading ROC/AUC for ${model}:`, err);
      }
    }
    setLoading(false);
    showMessage('success', `Loaded ${allModels.length} ROC/AUC curves`);
  };

  const loadAllGraphs = async (graphType = 'performance') => {
    if (!metrics) return;
    setLoading(true);
    const allModels = Object.keys(metrics);
    for (const model of allModels) {
      try {
        const url = `${API_URL}/visualization/graphs/${model}?type=${graphType}`;
        const result = await apiCall(url);
        setGraphImages(prev => ({ ...prev, [model + '_' + graphType]: result.image }));
      } catch (err) {
        console.error(`Error loading graphs for ${model}:`, err);
      }
    }
    setLoading(false);
    showMessage('success', `Loaded ${allModels.length} ${graphType} graphs`);
  };

  // ============ DATABASE FUNCTIONS ============
  const loadTrainingHistory = async () => {
    try {
      const data = await apiCall(`${ENDPOINTS.DB_TRAINING_HISTORY}?limit=10`);
      setTrainingHistory(data);
    } catch (err) {
      console.error('Error loading training history:', err);
    }
  };

  const loadPredictionHistory = async () => {
    try {
      const data = await apiCall(`${ENDPOINTS.DB_PREDICTION_HISTORY}?limit=50`);
      setPredictionHistory(data);
    } catch (err) {
      console.error('Error loading prediction history:', err);
    }
  };

  const loadModelComparison = async () => {
    try {
      const data = await apiCall(ENDPOINTS.DB_MODEL_COMPARISON);
      setModelComparison(data);
    } catch (err) {
      console.error('Error loading model comparison:', err);
    }
  };

  // ============ TABS CONFIGURATION ============
  const tabs = [
    { id: 'dataset', label: 'Dataset', icon: FaDatabase, component: DatasetTab },
    { id: 'training', label: 'Training', icon: FaBrain, component: TrainingTab },
    { id: 'prediction', label: 'Prediction', icon: FaChartLine, component: PredictionTab },
    { id: 'visualization', label: 'Analytics', icon: FaChartBar, component: VisualizationTab },
    { id: 'history', label: 'History', icon: FaHistory, component: HistoryTab },
    { id: 'report', label: 'Report', icon: FaFileAlt, component: ReportTab },
  ];

  // ============ RENDER ============
  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component;

  return (
    <div className="container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-title">
          <span>EduLens</span>
        </div>
        <nav>
          {tabs.map(tab => {
            const Icon = tab.icon;
            return (
              <button
                key={tab.id}
                className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.id)}
              >
                <Icon size={18} />
                <span>{tab.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-status">
          <FaCircle 
            size={10} 
            color={dbConnected ? '#3fb950' : '#f85149'} 
            style={{ marginRight: '4px' }}
          />
          <span>{dbConnected ? 'DB Connected' : 'DB Offline'}</span>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main">
        <header className="header">
          <h1>Academic Performance System</h1>
          <p>AI-Powered Student Performance Prediction</p>
        </header>

        {message.text && (
          <div className={`alert alert-${message.type}`}>
            {message.text}
          </div>
        )}

        {/* Render active tab component with props */}
        {ActiveComponent && (
          <ActiveComponent
            // Common props
            datasetInfo={datasetInfo}
            setDatasetInfo={setDatasetInfo}
            showMessage={showMessage}
            setLoading={setLoading}
            loading={loading}
            forceRefresh={forceRefresh}
            setModelsTrained={setModelsTrained}
            smotePreview={smotePreview}
            previewSmoteBalanced={previewSmoteBalanced}
            showSmotePreview={showSmotePreview}
            setShowSmotePreview={setShowSmotePreview}
            // Training props
            metrics={metrics}
            setMetrics={setMetrics}
            modelsTrained={modelsTrained}
            trainingStatus={trainingStatus}
            updateTrainingStatus={updateTrainingStatus}
            loadTrainingHistory={loadTrainingHistory}
            // Prediction props
            predictions={predictions}
            setPredictions={setPredictions}
            loadPredictionHistory={loadPredictionHistory}
            // Visualization props
            refreshTrigger={refreshTrigger}
            confusionMatrices={confusionMatrices}
            setConfusionMatrices={setConfusionMatrices}
            loadConfusionMatrix={loadConfusionMatrix}
            loadAllConfusionMatrices={loadAllConfusionMatrices}
            rocAucImages={rocAucImages}
            setRocAucImages={setRocAucImages}
            loadAllRocAuc={loadAllRocAuc}
            graphImages={graphImages}
            setGraphImages={setGraphImages}
            loadAllGraphs={loadAllGraphs}
            // History props
            trainingHistory={trainingHistory}
            predictionHistory={predictionHistory}
            modelComparison={modelComparison}
            loadModelComparison={loadModelComparison}
            dbConnected={dbConnected}
          />
        )}
      </main>
    </div>
  );
}

export default App;