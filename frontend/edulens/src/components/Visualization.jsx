import React, { useState, useEffect } from 'react';
import { 
  FaChartBar, 
  FaChartLine, 
  FaChartPie, 
  FaSync, 
  FaTrophy, 
  FaTable, 
  FaEye, 
  FaEyeSlash,
  FaSpinner,
  FaBrain,
  FaProjectDiagram,
  FaNetworkWired
} from 'react-icons/fa';
import { API_URL, ENDPOINTS, apiCall } from '../utils/api';

function Visualization({
  metrics,
  showMessage,
  modelsTrained,
  refreshTrigger,
  forceRefresh,
  confusionMatrices,
  setConfusionMatrices,
  loadConfusionMatrix,
  loadAllConfusionMatrices,
  rocAucImages,
  setRocAucImages,
  loadAllRocAuc,
  graphImages,
  setGraphImages,
  loadAllGraphs,
}) {
  // ============ STATE ============
  const [activeViz, setActiveViz] = useState('comparison');
  const [loading, setLoading] = useState(false);
  const [images, setImages] = useState({});
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [confusionLoading, setConfusionLoading] = useState(false);
  const [activeGraphType, setActiveGraphType] = useState('performance');
  const [bestModelMatrix, setBestModelMatrix] = useState(null);
  const [bestModelRoc, setBestModelRoc] = useState(null);
  const [bestModelName, setBestModelName] = useState('');
  const [loadingBestModel, setLoadingBestModel] = useState(false);
  const [graphLoading, setGraphLoading] = useState(false);

  // ============ CONSTANTS ============
  const modelDisplayNames = {
    'knn_euclidean': 'KNN (Euclidean)',
    'knn_manhattan': 'KNN (Manhattan)',
    'multinomial_nb': 'Multinomial NB',
    'svm_rbf': 'RBF SVM',
    'neuralnetwork_softmax': 'NN Softmax',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
  };

  // ============ VISUALIZATION LOADING ============
  const loadVisualization = async (type, force = false) => {
    if (!modelsTrained) return;
    setLoading(true);
    try {
      let url;
      if (type === 'comparison') url = ENDPOINTS.VIZ_MODEL_COMPARE;
      else if (type === 'distribution') url = ENDPOINTS.VIZ_CLASS_DIST;
      else if (type === 'correlation') url = ENDPOINTS.VIZ_FEATURE_CORR;

      const fetchUrl = force ? `${url}?t=${Date.now()}` : url;
      const result = await apiCall(fetchUrl);

      setImages(prev => ({ ...prev, [type]: result.image }));
      setLastRefresh(new Date().toLocaleTimeString());
      if (force) showMessage('success', 'Visualization refreshed');
    } catch (err) {
      showMessage('error', 'Error loading visualization');
    } finally {
      setLoading(false);
    }
  };

  // ============ LOAD GRAPHS FOR SPECIFIC MODEL ============
  const loadGraphForModel = async (modelName, graphType = 'performance') => {
    if (!modelsTrained || !modelName) return;
    setGraphLoading(true);
    try {
      const url = `${API_URL}/visualization/graphs/${modelName}?type=${graphType}`;
      const result = await apiCall(url);
      setGraphImages(prev => ({ 
        ...prev, 
        [`${modelName}_${graphType}`]: result.image 
      }));
      showMessage('success', `Loaded ${graphType} graph for ${modelDisplayNames[modelName] || modelName}`);
    } catch (err) {
      console.error(`Error loading graph for ${modelName}:`, err);
      showMessage('error', `Failed to load ${graphType} graph`);
    } finally {
      setGraphLoading(false);
    }
  };

  // ============ LOAD ALL GRAPHS ============
  const loadAllGraphsForType = async (graphType = 'performance') => {
    if (!modelsTrained) return;
    setGraphLoading(true);
    const allModels = Object.keys(metrics || {});
    let successCount = 0;
    
    for (const model of allModels) {
      try {
        const url = `${API_URL}/visualization/graphs/${model}?type=${graphType}`;
        const result = await apiCall(url);
        setGraphImages(prev => ({ 
          ...prev, 
          [`${model}_${graphType}`]: result.image 
        }));
        successCount++;
      } catch (err) {
        console.error(`Error loading graph for ${model}:`, err);
      }
    }
    
    setGraphLoading(false);
    if (successCount > 0) {
      showMessage('success', `Loaded ${successCount} ${graphType} graphs`);
    } else {
      showMessage('error', 'Failed to load graphs');
    }
  };

  // ============ HANDLE GRAPH BUTTON CLICKS ============
  const handleGraphTypeChange = (graphType) => {
    setActiveGraphType(graphType);
    const allModels = Object.keys(metrics || {});
    const hasGraphs = allModels.some(model => graphImages[`${model}_${graphType}`]);
    
    if (!hasGraphs) {
      loadAllGraphsForType(graphType);
    }
  };

  // ============ BEST MODEL VISUALIZATIONS ============
  const loadBestModelMatrix = async () => {
    if (!modelsTrained) return;
    setLoadingBestModel(true);
    try {
      const result = await apiCall(ENDPOINTS.VIZ_BEST_MATRIX);
      setBestModelMatrix(result.image);
      setBestModelName(result.model_name || 'Best Model');
      showMessage('success', 'Best model confusion matrix loaded');
    } catch (err) {
      showMessage('error', 'Failed to load best model matrix');
    } finally {
      setLoadingBestModel(false);
    }
  };

  const loadBestModelRoc = async () => {
    if (!modelsTrained) return;
    setLoadingBestModel(true);
    try {
      const result = await apiCall(ENDPOINTS.VIZ_BEST_ROC);
      setBestModelRoc(result.image);
      setBestModelName(result.model_name || 'Best Model');
      showMessage('success', 'Best model ROC curve loaded');
    } catch (err) {
      showMessage('error', 'Failed to load best model ROC');
    } finally {
      setLoadingBestModel(false);
    }
  };

  // ============ LOAD ALL CONFUSION MATRICES ============
  const loadAllConfusionMatricesLocal = async () => {
    if (!modelsTrained) return;
    setConfusionLoading(true);
    const allModels = Object.keys(metrics || {});
    let successCount = 0;
    
    for (const model of allModels) {
      try {
        const url = `${API_URL}/visualization/confusion_matrix/${model}`;
        const result = await apiCall(url);
        setConfusionMatrices(prev => ({ ...prev, [model]: result.image }));
        successCount++;
      } catch (err) {
        console.error(`Error loading ${model}:`, err);
      }
    }
    setConfusionLoading(false);
    if (successCount > 0) {
      showMessage('success', `Loaded ${successCount} confusion matrices`);
    }
  };

  // ============ LOAD ALL ROC CURVES ============
  const loadAllRocAucLocal = async () => {
    if (!modelsTrained) return;
    setConfusionLoading(true);
    const allModels = Object.keys(metrics || {});
    let successCount = 0;
    
    for (const model of allModels) {
      try {
        const url = `${API_URL}/visualization/roc_auc/${model}`;
        const result = await apiCall(url);
        setRocAucImages(prev => ({ ...prev, [model]: result.image }));
        successCount++;
      } catch (err) {
        console.error(`Error loading ROC for ${model}:`, err);
      }
    }
    setConfusionLoading(false);
    if (successCount > 0) {
      showMessage('success', `Loaded ${successCount} ROC curves`);
    }
  };

  // ============ HELPERS ============
  const findBestModel = () => {
    if (!metrics) return null;
    let best = null;
    let bestAccuracy = -1;
    for (const model in metrics) {
      if (metrics[model] && metrics[model].accuracy > bestAccuracy) {
        bestAccuracy = metrics[model].accuracy;
        best = model;
      }
    }
    return best;
  };

  // ============ EFFECTS ============
  useEffect(() => {
    let interval;
    if (autoRefresh && modelsTrained && activeViz) {
      interval = setInterval(() => loadVisualization(activeViz, true), 10000);
    }
    return () => { if (interval) clearInterval(interval); };
  }, [autoRefresh, modelsTrained, activeViz]);

  useEffect(() => {
    if (modelsTrained && activeViz && (!images[activeViz] || refreshTrigger)) {
      loadVisualization(activeViz);
    }
  }, [activeViz, modelsTrained, refreshTrigger]);

  const bestModelFromMetrics = findBestModel();

  // ============ RENDER ============
  return (
    <div className="fade-in">
      {/* Main Visualizations */}
      <div className="card">
        <div className="card-title" style={{ fontSize: '16px', marginBottom: '12px' }}>
          <FaChartLine size={18} />
          <span>Data Visualizations</span>
        </div>

        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', padding: '8px 12px', background: '#161b22', borderRadius: '8px', flexWrap: 'wrap', gap: '8px' }}>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button 
              className="btn-secondary" 
              onClick={() => loadVisualization(activeViz, true)} 
              disabled={loading || !modelsTrained}
              style={{ padding: '4px 12px', fontSize: '12px' }}
            >
              <FaSync className={loading ? 'spinner' : ''} size={12} /> {loading ? 'Loading...' : 'Refresh'}
            </button>
            <button
              className={autoRefresh ? 'btn-danger' : 'btn-secondary'}
              onClick={() => {
                setAutoRefresh(!autoRefresh);
                showMessage('info', autoRefresh ? 'Auto-refresh OFF' : 'Auto-refresh ON');
              }}
              disabled={!modelsTrained}
              style={{ padding: '4px 12px', fontSize: '12px' }}
            >
              {autoRefresh ? '⏸️ Auto ON' : '▶️ Auto OFF'}
            </button>
          </div>
          {lastRefresh && <span style={{ fontSize: '10px', color: '#8b949e' }}>Last: {lastRefresh}</span>}
        </div>

        <div className="tabs" style={{ marginBottom: '12px' }}>
          <button
            className={`tab ${activeViz === 'comparison' ? 'active' : ''}`}
            onClick={() => setActiveViz('comparison')}
            style={{ padding: '6px 14px', fontSize: '12px' }}
          >
            <FaChartBar size={14} /> Model Comparison
          </button>
          <button
            className={`tab ${activeViz === 'distribution' ? 'active' : ''}`}
            onClick={() => setActiveViz('distribution')}
            style={{ padding: '6px 14px', fontSize: '12px' }}
          >
            <FaChartPie size={14} /> Grade Distribution
          </button>
          <button
            className={`tab ${activeViz === 'correlation' ? 'active' : ''}`}
            onClick={() => setActiveViz('correlation')}
            style={{ padding: '6px 14px', fontSize: '12px' }}
          >
            <FaNetworkWired size={14} /> Feature Correlation
          </button>
        </div>

        {!modelsTrained ? (
          <div className="empty-state" style={{ padding: '40px' }}>
            <FaChartBar size={40} style={{ color: '#484f58', marginBottom: '12px' }} />
            <h3 style={{ fontSize: '16px' }}>No Models Trained</h3>
            <p style={{ fontSize: '13px' }}>Please train models first to see visualizations</p>
          </div>
        ) : loading ? (
          <div style={{ textAlign: 'center', padding: '40px' }}>
            <FaSpinner className="spinner" size={28} />
            <p style={{ marginTop: '12px', color: '#8b949e', fontSize: '13px' }}>Loading visualization...</p>
          </div>
        ) : (
          images[activeViz] && <img src={images[activeViz]} className="viz-image" style={{ maxHeight: '400px', objectFit: 'contain' }} />
        )}
      </div>

      {/* Best Model Analysis */}
      {modelsTrained && (
        <div className="card">
          <div className="card-title" style={{ fontSize: '16px', marginBottom: '12px' }}>
            <FaTrophy size={18} />
            <span>Best Model Analysis</span>
          </div>

          <div style={{ marginBottom: '12px', padding: '8px 12px', background: 'rgba(63, 185, 80, 0.1)', borderRadius: '8px', fontSize: '13px' }}>
            <span>Best Model: </span>
            <strong>{modelDisplayNames[bestModelFromMetrics] || bestModelFromMetrics}</strong>
            <span> (Accuracy: {((metrics && metrics[bestModelFromMetrics]?.accuracy) * 100).toFixed(2)}%)</span>
          </div>

          <div style={{ display: 'flex', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
            <button 
              className="btn-primary" 
              onClick={loadBestModelMatrix} 
              disabled={loadingBestModel || !modelsTrained}
              style={{ padding: '4px 12px', fontSize: '12px' }}
            >
              {loadingBestModel ? <FaSpinner className="spinner" size={12} /> : <FaTable size={14} />} Show Confusion Matrix
            </button>
            <button 
              className="btn-secondary" 
              onClick={loadBestModelRoc} 
              disabled={loadingBestModel || !modelsTrained}
              style={{ padding: '4px 12px', fontSize: '12px' }}
            >
              {loadingBestModel ? <FaSpinner className="spinner" size={12} /> : <FaChartLine size={14} />} Show ROC Curve
            </button>
          </div>

          {bestModelMatrix && (
            <div style={{ marginTop: '12px' }}>
              <h4 style={{ marginBottom: '8px', fontSize: '14px', color: '#e6edf3' }}>Confusion Matrix</h4>
              <img src={bestModelMatrix} className="viz-image" style={{ maxHeight: '400px', objectFit: 'contain' }} />
            </div>
          )}

          {bestModelRoc && (
            <div style={{ marginTop: '12px' }}>
              <h4 style={{ marginBottom: '8px', fontSize: '14px', color: '#e6edf3' }}>ROC/AUC Curve</h4>
              <img src={bestModelRoc} className="viz-image" style={{ maxHeight: '400px', objectFit: 'contain' }} />
            </div>
          )}
        </div>
      )}

      {/* Confusion Matrices */}
      {modelsTrained && (
        <div className="card">
          <div className="card-title" style={{ fontSize: '16px', marginBottom: '12px' }}>
            <FaTable size={18} />
            <span>Confusion Matrices</span>
          </div>

          <button 
            className="btn-primary" 
            onClick={loadAllConfusionMatricesLocal} 
            disabled={confusionLoading}
            style={{ padding: '4px 12px', fontSize: '12px' }}
          >
            {confusionLoading ? <FaSpinner className="spinner" size={12} /> : <FaEye size={14} />} Load All Confusion Matrices
          </button>

          {confusionLoading && (
            <div style={{ textAlign: 'center', padding: '16px' }}>
              <FaSpinner className="spinner" size={20} />
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px', marginTop: '12px' }}>
            {Object.entries(confusionMatrices).map(([model, img]) => {
              const accuracy = metrics && metrics[model] ? (metrics[model].accuracy * 100).toFixed(2) : 'N/A';
              return img && (
                <div key={model} style={{ padding: '12px', background: '#161b22', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '13px' }}>
                    <h4 style={{ margin: 0, fontSize: '14px' }}>{modelDisplayNames[model] || model}</h4>
                    <span style={{ color: '#3fb950', fontWeight: '600', fontSize: '12px' }}>Acc: {accuracy}%</span>
                  </div>
                  <img src={img} style={{ width: '100%', borderRadius: '6px', maxHeight: '350px', objectFit: 'contain' }} />
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* ROC/AUC Curves */}
      {modelsTrained && (
        <div className="card">
          <div className="card-title" style={{ fontSize: '16px', marginBottom: '12px' }}>
            <FaChartLine size={18} />
            <span>ROC/AUC Curves</span>
          </div>

          <button 
            className="btn-primary" 
            onClick={loadAllRocAucLocal} 
            disabled={confusionLoading}
            style={{ padding: '4px 12px', fontSize: '12px' }}
          >
            {confusionLoading ? <FaSpinner className="spinner" size={12} /> : <FaEye size={14} />} Load All ROC Curves
          </button>

          {confusionLoading && (
            <div style={{ textAlign: 'center', padding: '16px' }}>
              <FaSpinner className="spinner" size={20} />
            </div>
          )}

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px', marginTop: '12px' }}>
            {Object.entries(rocAucImages).map(([model, img]) => (
              img && (
                <div key={model} style={{ padding: '12px', background: '#161b22', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                  <h4 style={{ marginBottom: '8px', fontSize: '14px' }}>{modelDisplayNames[model] || model}</h4>
                  <img src={img} style={{ width: '100%', borderRadius: '6px', maxHeight: '350px', objectFit: 'contain' }} />
                </div>
              )
            ))}
          </div>
        </div>
      )}

      {/* Model Performance Graphs */}
      {modelsTrained && (
        <div className="card">
          <div className="card-title" style={{ fontSize: '16px', marginBottom: '12px' }}>
            <FaBrain size={18} />
            <span>Model Performance Graphs</span>
          </div>

          <div style={{ display: 'flex', gap: '6px', marginBottom: '12px', flexWrap: 'wrap' }}>
            <button
              className={activeGraphType === 'performance' ? 'btn-primary' : 'btn-secondary'}
              onClick={() => handleGraphTypeChange('performance')}
              disabled={graphLoading}
              style={{ padding: '4px 12px', fontSize: '12px' }}
            >
              {graphLoading && activeGraphType === 'performance' ? <FaSpinner className="spinner" size={12} /> : <FaChartBar size={14} />} 
              Performance
            </button>
            <button
              className={activeGraphType === 'feature' ? 'btn-primary' : 'btn-secondary'}
              onClick={() => handleGraphTypeChange('feature')}
              disabled={graphLoading}
              style={{ padding: '4px 12px', fontSize: '12px' }}
            >
              {graphLoading && activeGraphType === 'feature' ? <FaSpinner className="spinner" size={12} /> : <FaProjectDiagram size={14} />} 
              Feature Importance
            </button>
            <button
              className={activeGraphType === 'learning' ? 'btn-primary' : 'btn-secondary'}
              onClick={() => handleGraphTypeChange('learning')}
              disabled={graphLoading}
              style={{ padding: '4px 12px', fontSize: '12px' }}
            >
              {graphLoading && activeGraphType === 'learning' ? <FaSpinner className="spinner" size={12} /> : <FaChartLine size={14} />} 
              Learning Curves
            </button>
          </div>

          {graphLoading ? (
            <div style={{ textAlign: 'center', padding: '30px' }}>
              <FaSpinner className="spinner" size={28} />
              <p style={{ marginTop: '12px', color: '#8b949e', fontSize: '13px' }}>Loading {activeGraphType} graphs...</p>
            </div>
          ) : (
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '16px' }}>
              {Object.entries(graphImages)
                .filter(([key]) => key.includes(activeGraphType))
                .map(([key, img]) => {
                  const model = key.replace(`_${activeGraphType}`, '');
                  return img && (
                    <div key={key} style={{ padding: '12px', background: '#161b22', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                      <h4 style={{ marginBottom: '8px', fontSize: '14px' }}>{modelDisplayNames[model] || model}</h4>
                      <img src={img} style={{ width: '100%', borderRadius: '6px', maxHeight: '350px', objectFit: 'contain' }} />
                    </div>
                  );
                })}
            </div>
          )}

          {Object.keys(graphImages).filter(key => key.includes(activeGraphType)).length === 0 && !graphLoading && (
            <div className="empty-state" style={{ padding: '30px' }}>
              <p style={{ fontSize: '13px' }}>No {activeGraphType} graphs loaded. Click a button above to load graphs.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default Visualization;