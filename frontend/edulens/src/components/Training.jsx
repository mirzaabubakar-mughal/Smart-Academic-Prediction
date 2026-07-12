import React, { useState } from 'react';
import { 
  FaBrain, 
  FaRocket, 
  FaTrash, 
  FaCheckCircle, 
  FaExclamationTriangle, 
  FaTrophy, 
  FaSpinner,
  FaChartBar,
  FaClock
} from 'react-icons/fa';
import { ENDPOINTS, apiCall } from '../utils/api';

// Metric Row Component
const MetricRow = ({ label, value }) => (
  <div className="metric-row">
    <div className="metric-row-header">
      <span>{label}</span>
      <span>{(value * 100).toFixed(2)}%</span>
    </div>
    <div className="metric-bar">
      <div className="metric-bar-fill" style={{ width: `${value * 100}%` }} />
    </div>
  </div>
);

function Training({
  metrics,
  setMetrics,
  setDatasetInfo,
  showMessage,
  setLoading,
  loading,
  modelsTrained,
  setModelsTrained,
  forceRefresh,
  trainingStatus,
  updateTrainingStatus,
  loadTrainingHistory,
}) {
  // ============ STATE ============
  const [progress, setProgress] = useState(0);
  const [currentModel, setCurrentModel] = useState('');
  const [trainingComplete, setTrainingComplete] = useState(false);

  // ============ CONSTANTS ============
  const allModelsList = [
    'KNN (Euclidean Distance)',
    'KNN (Manhattan Distance)',
    'Multinomial Naive Bayes',
    'Neural Network (Softmax)',
    'Decision Tree (CART)',
    'Random Forest (Ensemble)',
    'Logistic Regression',
  ];

  const modelDisplayNames = {
    'knn_euclidean': 'KNN (Euclidean Distance)',
    'knn_manhattan': 'KNN (Manhattan Distance)',
    'multinomial_nb': 'Multinomial Naive Bayes',
    'svm_rbf': 'Non-Linear SVM (RBF Kernel)',
    'neuralnetwork_softmax': 'Neural Network (Softmax)',
    'decision_tree': 'Decision Tree (CART)',
    'random_forest': 'Random Forest (Ensemble)',
    'logistic_regression': 'Logistic Regression',
  };

  // ============ HANDLERS ============
  const handleTrain = async () => {
    setLoading(true);
    updateTrainingStatus('training');
    setProgress(0);
    setTrainingComplete(false);

    try {
      // Simulate progress
      for (let i = 0; i < allModelsList.length; i++) {
        setCurrentModel(allModelsList[i]);
        setProgress(((i + 1) / allModelsList.length) * 100);
        await new Promise(resolve => setTimeout(resolve, 200));
      }

      // API call
      const result = await apiCall(ENDPOINTS.TRAIN, { method: 'POST' });
      
      setMetrics(result.metrics);
      if (result.dataset_info) setDatasetInfo(result.dataset_info);
      
      setProgress(100);
      updateTrainingStatus('completed', true);
      setTrainingComplete(true);
      
      showMessage(
        'success',
        `${result.best_model} achieved ${(result.best_accuracy * 100).toFixed(2)}% accuracy!`
      );

      if (loadTrainingHistory) loadTrainingHistory();
      forceRefresh();
    } catch (err) {
      console.error('Training error:', err);
      updateTrainingStatus('failed');
      setTrainingComplete(false);
      showMessage('error', err.message || 'Error training models.');
    } finally {
      setLoading(false);
      setCurrentModel('');
      setTimeout(() => setProgress(0), 1000);
    }
  };

  const clearResults = () => {
    setMetrics(null);
    setModelsTrained(false);
    updateTrainingStatus('idle');
    setTrainingComplete(false);
    showMessage('info', 'Training results cleared');
    forceRefresh();
  };

  // ============ HELPERS ============
  const getBestModel = () => {
    if (!metrics) return null;
    return Object.entries(metrics).reduce(
      (best, [name, m]) => (!best || m.accuracy > metrics[best]?.accuracy) ? name : best,
      null
    );
  };

  const bestModel = getBestModel();

  // ============ RENDER ============
  return (
    <div className="fade-in">
      {/* Training Control Card */}
      <div className="card">
        <div className="card-title">
          <FaBrain />
          <span>Train ML Models</span>
        </div>
        <p className="card-subtitle">
          Training {allModelsList.length} supervised learning classification algorithms on your dataset
        </p>

        {/* Training Progress */}
        {trainingStatus === 'training' && (
          <div className="training-progress" style={{ marginBottom: '20px', padding: '16px', background: 'rgba(88, 166, 255, 0.05)', borderRadius: '12px', border: '1px solid #58a6ff' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px', fontSize: '14px', fontWeight: '600', color: '#58a6ff' }}>
              <FaSpinner className="spinner" />
              <span>Training {currentModel}... {Math.floor(progress)}%</span>
            </div>
            <div style={{ width: '100%', height: '10px', background: '#161b22', borderRadius: '5px', overflow: 'hidden' }}>
              <div style={{ width: `${progress}%`, height: '100%', background: 'linear-gradient(90deg, #58a6ff, #a371f7)', transition: 'width 0.3s ease' }} />
            </div>
            <div style={{ marginTop: '8px', fontSize: '12px', color: '#8b949e', textAlign: 'center' }}>
              Model {Math.floor(progress / (100 / allModelsList.length)) + 1} of {allModelsList.length}
            </div>
          </div>
        )}

        {/* Action Buttons */}
        <div className="card-actions">
          <button className="btn-primary" onClick={handleTrain} disabled={loading}>
            {loading ? (
              <>
                <FaSpinner className="spinner" /> Training {Math.floor(progress)}%...
              </>
            ) : (
              <>
                <FaRocket /> Start Training ({allModelsList.length} Algorithms)
              </>
            )}
          </button>
          {metrics && (
            <button className="btn-secondary" onClick={clearResults} disabled={loading}>
              <FaTrash /> Clear Results
            </button>
          )}
        </div>

        {/* Status Messages */}
        {trainingStatus === 'completed' && trainingComplete && (
          <div className="alert alert-success" style={{ marginTop: '16px' }}>
            <FaCheckCircle /> Training Complete! All {allModelsList.length} models are ready.
          </div>
        )}
        {trainingStatus === 'failed' && (
          <div className="alert alert-error" style={{ marginTop: '16px' }}>
            <FaExclamationTriangle /> Training Failed. Please check dataset.
          </div>
        )}
      </div>

      {/* Model Performance Metrics */}
      {metrics && Object.keys(metrics).length > 0 && (
        <div className="card fade-in">
          <div className="card-title">
            <FaChartBar />
            <span>Model Performance Metrics ({Object.keys(metrics).length} Algorithms)</span>
          </div>

          <div className="metrics-grid" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '16px', maxHeight: '600px', overflowY: 'auto' }}>
            {Object.entries(metrics)
              .sort((a, b) => b[1].accuracy - a[1].accuracy)
              .map(([model, m]) => {
                const isBest = model === bestModel;
                return (
                  <div
                    key={model}
                    className="metric-card"
                    style={{
                      padding: '16px',
                      borderRadius: '12px',
                      background: isBest ? 'rgba(63, 185, 80, 0.05)' : 'var(--bg-secondary)',
                      border: `1px solid ${isBest ? '#3fb950' : 'var(--border-color)'}`,
                    }}
                  >
                    <div className="metric-card-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                      <span style={{ fontWeight: '600', color: '#e6edf3' }}>{modelDisplayNames[model] || model}</span>
                      {isBest && (
                        <span style={{ background: '#3fb950', color: '#0d1117', padding: '2px 8px', borderRadius: '20px', fontSize: '10px', display: 'flex', alignItems: 'center', gap: '4px' }}>
                          <FaTrophy /> BEST
                        </span>
                      )}
                    </div>
                    <MetricRow label="Accuracy" value={m.accuracy} />
                    <MetricRow label="Precision" value={m.precision} />
                    <MetricRow label="Recall" value={m.recall} />
                    <MetricRow label="F1-Score" value={m.f1} />
                  </div>
                );
              })}
          </div>

          {/* Best Model Summary */}
          {bestModel && (
            <div style={{ marginTop: '20px', padding: '16px', background: 'var(--bg-secondary)', borderRadius: '12px', border: '1px solid var(--border-color)', textAlign: 'center' }}>
              <div style={{ fontWeight: '600', color: '#8b949e', marginBottom: '4px' }}>
                <FaTrophy /> Overall Best Model
              </div>
              <div style={{ fontSize: '18px', fontWeight: '700', color: '#e6edf3' }}>
                {modelDisplayNames[bestModel] || bestModel}
              </div>
              <div style={{ color: '#3fb950', fontSize: '14px' }}>
                Accuracy: {(metrics[bestModel].accuracy * 100).toFixed(2)}%
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Training;