import React, { useState } from 'react';
import { 
  FaDatabase, 
  FaTrash, 
  FaChartBar, 
  FaCube, 
  FaChartLine, 
  FaSync,
  FaTable,
  FaBalanceScale,
  FaEye,
  FaEyeSlash
} from 'react-icons/fa';
import { ENDPOINTS, apiCall } from '../utils/api';

function Dataset({
  datasetInfo,
  setDatasetInfo,
  showMessage,
  setLoading,
  loading,
  forceRefresh,
  setModelsTrained,
  showSmotePreview,
  setShowSmotePreview,
}) {
  // ============ STATE ============
  const [dataLoaded, setDataLoaded] = useState(false);
  const [dataPreview, setDataPreview] = useState(null);
  const [classDistribution, setClassDistribution] = useState(null);
  const [imbalanceReport, setImbalanceReport] = useState(null);
  const [pcaImage, setPcaImage] = useState(null);
  const [marks2DImage, setMarks2DImage] = useState(null);
  const [features3DImage, setFeatures3DImage] = useState(null);
  const [loadingViz, setLoadingViz] = useState(false);
  const [activeViz, setActiveViz] = useState('2d');
  const [smoteData, setSmoteData] = useState(null);
  const [smoteLoading, setSmoteLoading] = useState(false);

  // ============ HELPERS ============
  const getGradeColor = (grade) => {
    const colors = { A: '#3fb950', B: '#58a6ff', C: '#d29922', D: '#f0883e', F: '#f85149' };
    return colors[grade] || '#8b949e';
  };

  // ============ DATA LOADING ============
  const loadDefaultDataset = async () => {
    setLoading(true);
    setShowSmotePreview(false);
    setSmoteData(null);

    try {
      const previewData = await apiCall(ENDPOINTS.DATASET_PREVIEW);
      const infoData = await apiCall(ENDPOINTS.DATASET_INFO);

      if (Array.isArray(previewData) && previewData.length > 0) {
        setDataPreview(previewData);
        analyzeClassDistribution(previewData);
        loadAllVisualizations();
      }

      setDatasetInfo(infoData);
      setDataLoaded(true);
      showMessage('success', 'Synthetic dataset loaded successfully!');
      forceRefresh();
    } catch (err) {
      console.error('Error loading dataset:', err);
      showMessage('error', err.message || 'Error loading dataset');
      setDataPreview(null);
      setDatasetInfo(null);
      setDataLoaded(false);
    } finally {
      setLoading(false);
    }
  };

  // ============ SMOTE ============
  const loadSmotePreview = async () => {
    setSmoteLoading(true);
    setShowSmotePreview(false);
    try {
      const result = await apiCall(ENDPOINTS.DATASET_SMOTE);
      setSmoteData(result);
      setShowSmotePreview(true);
      showMessage('success', 'SMOTE balanced dataset preview loaded!');
    } catch (err) {
      console.error('SMOTE preview error:', err);
      showMessage('error', err.message || 'Error loading SMOTE preview');
      setShowSmotePreview(false);
    } finally {
      setSmoteLoading(false);
    }
  };

  // ============ VISUALIZATIONS ============
  const loadAllVisualizations = async () => {
    setLoadingViz(true);
    try {
      const [marks2D, features3D, pca] = await Promise.all([
        apiCall(ENDPOINTS.VIZ_2D),
        apiCall(ENDPOINTS.VIZ_3D),
        apiCall(ENDPOINTS.VIZ_PCA),
      ]);
      setMarks2DImage(marks2D.image);
      setFeatures3DImage(features3D.image);
      setPcaImage(pca.image);
    } catch (err) {
      console.error('Error loading visualizations:', err);
    } finally {
      setLoadingViz(false);
    }
  };

  // ============ CLASS DISTRIBUTION ============
  const analyzeClassDistribution = (data) => {
    if (!data || !Array.isArray(data) || data.length === 0) return;

    const firstRow = data[0];
    let gradeColumn = null;
    for (const key in firstRow) {
      if (key.toLowerCase().includes('grade') || key === 'grade') {
        gradeColumn = key;
        break;
      }
    }
    if (!gradeColumn) return;

    const classCounts = {};
    for (let i = 0; i < data.length; i++) {
      const grade = data[i][gradeColumn];
      if (grade) classCounts[grade] = (classCounts[grade] || 0) + 1;
    }

    const total = data.length;
    const classes = Object.keys(classCounts).sort();
    const distribution = [];
    let maxCount = 0, minCount = Infinity, majorityClass = '', minorityClass = '';

    for (const cls of classes) {
      const count = classCounts[cls];
      const percentage = (count / total) * 100;
      distribution.push({ class: cls, count, percentage });
      if (count > maxCount) { maxCount = count; majorityClass = cls; }
      if (count < minCount) { minCount = count; minorityClass = cls; }
    }

    const imbalanceRatio = (maxCount / minCount).toFixed(2);
    const isImbalanced = imbalanceRatio > 1.5;

    setClassDistribution(distribution);
    setImbalanceReport({
      totalSamples: total,
      majorityClass,
      majorityCount: maxCount,
      minorityClass,
      minorityCount: minCount,
      imbalanceRatio,
      isImbalanced,
    });
  };

  // ============ CLEAR DATA ============
  const clearData = () => {
    setDataPreview(null);
    setDatasetInfo(null);
    setDataLoaded(false);
    setModelsTrained(false);
    setClassDistribution(null);
    setImbalanceReport(null);
    setPcaImage(null);
    setMarks2DImage(null);
    setFeatures3DImage(null);
    setShowSmotePreview(false);
    setSmoteData(null);
    showMessage('info', 'Dataset cleared successfully');
    forceRefresh();
  };

  // ============ GET DATA NATURE ============
  const getDataNature = () => {
    if (!datasetInfo || !datasetInfo.columns) return null;
    const columns = datasetInfo.columns;
    const dtypes = datasetInfo.dtypes || {};
    const targetCol = columns.find(c => c.toLowerCase().includes('grade')) || 'grade';

    const targetUnique = dataPreview
      ? [...new Set(dataPreview.map(row => row[targetCol]))]
      : [];

    let problemType = 'Classification';
    if (targetUnique.length <= 10 && targetUnique.length > 2) {
      problemType = 'Multi-class Classification';
    } else if (targetUnique.length === 2) {
      problemType = 'Binary Classification';
    }

    return {
      problemType,
      totalSamples: datasetInfo.shape?.[0] || 0,
      totalFeatures: datasetInfo.shape?.[1] || 0,
      targetColumn: targetCol,
      uniqueClasses: targetUnique.length,
      missingValues: Object.values(datasetInfo.missing || {}).reduce((a, b) => a + b, 0),
    };
  };

  const dataNature = getDataNature();
  const hasData = dataPreview && Array.isArray(dataPreview) && dataPreview.length > 0;
  const headers = hasData ? Object.keys(dataPreview[0]) : [];

  // ============ RENDER ============
  return (
    <div className="fade-in">
      {/* Load Dataset Card */}
      <div className="card">
        <div className="card-title">
          <FaDatabase />
          <span>Load Dataset</span>
        </div>
        <div className="card-actions">
          <button className="btn-secondary" onClick={loadDefaultDataset} disabled={loading}>
            <FaDatabase /> Load Synthetic Dataset
          </button>
          {dataLoaded && (
            <button className="btn-danger" onClick={clearData} disabled={loading}>
              <FaTrash /> Clear Data
            </button>
          )}
        </div>
      </div>

      {/* Dataset Nature Card */}
      {datasetInfo && dataNature && (
        <div className="card dataset-nature">
          <div className="card-title">
            <FaChartBar />
            <span>Dataset Nature</span>
          </div>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value"><h6>{dataNature.problemType}</h6></div>
              <div className="stat-label">Problem Type</div>
            </div>
            <div className="stat-card">
              <div className="stat-value"><h6>{dataNature.targetColumn}</h6></div>
              <div className="stat-label">Target Column</div>
            </div>
            <div className="stat-card">
              <div className="stat-value"><h6>{dataNature.uniqueClasses}</h6></div>
              <div className="stat-label">Number of Classes</div>
            </div>
          </div>
          {showSmotePreview && smoteData && (
            <div className="alert alert-success" style={{ marginTop: '12px' }}>
              After SMOTE: {smoteData.balanced_samples} samples (Added {smoteData.synthetic_added} synthetic samples)
            </div>
          )}
        </div>
      )}

      {/* Statistics Cards */}
      {datasetInfo && (
        <div className="stats-grid">
          <div className="stat-card">
            <div className="stat-value">{(datasetInfo.shape && datasetInfo.shape[0]) || 0}</div>
            <div className="stat-label">Total Samples</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{(datasetInfo.shape && datasetInfo.shape[1]) || 0}</div>
            <div className="stat-label">Total Features</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{(dataNature && dataNature.uniqueClasses) || 0}</div>
            <div className="stat-label">Grade Classes</div>
          </div>
          <div className="stat-card">
            <div className="stat-value">{(dataNature && dataNature.missingValues) || 0}</div>
            <div className="stat-label">Missing Values</div>
          </div>
        </div>
      )}

      {/* Class Distribution */}
      {classDistribution && imbalanceReport && (
        <div className="card fade-in">
          <div className="card-title">
            <FaBalanceScale />
            <span>Class Distribution & Imbalance Analysis</span>
          </div>

          <div className="distribution-bars">
            <h4 style={{ marginBottom: '12px', fontSize: '14px' }}>Grade Distribution</h4>
            {classDistribution.map(item => (
              <div key={item.class} style={{ marginBottom: '12px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px' }}>
                  <span>Grade {item.class}</span>
                  <span>{item.count} samples ({item.percentage.toFixed(1)}%)</span>
                </div>
                <div style={{ width: '100%', height: '28px', background: '#161b22', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${item.percentage}%`,
                      height: '100%',
                      background: getGradeColor(item.class),
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'flex-end',
                      paddingRight: '8px',
                      color: 'white',
                      fontSize: '12px',
                      fontWeight: '600'
                    }}
                  >
                    {item.percentage > 15 && Math.floor(item.percentage) + '%'}
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div
            className="card"
            style={{
              background: imbalanceReport.isImbalanced ? 'rgba(210, 153, 34, 0.1)' : 'rgba(63, 185, 80, 0.1)',
              border: 'none',
              padding: '16px'
            }}
          >
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', gap: '12px' }}>
              <div>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#8b949e' }}>Majority Class</div>
                <div style={{ fontSize: '20px', fontWeight: '700' }}>{imbalanceReport.majorityClass}</div>
                <div style={{ fontSize: '12px', color: '#8b949e' }}>{imbalanceReport.majorityCount} samples</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#8b949e' }}>Minority Class</div>
                <div style={{ fontSize: '20px', fontWeight: '700' }}>{imbalanceReport.minorityClass}</div>
                <div style={{ fontSize: '12px', color: '#8b949e' }}>{imbalanceReport.minorityCount} samples</div>
              </div>
              <div>
                <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#8b949e' }}>Imbalance Ratio</div>
                <div style={{ fontSize: '20px', fontWeight: '700' }}>1:{imbalanceReport.imbalanceRatio}</div>
              </div>
            </div>
          </div>

          {imbalanceReport.isImbalanced ? (
            <button className="btn-primary full-width" style={{ justifyContent: 'center', background: 'linear-gradient(135deg, var(--accent), #a371f7)' }} onClick={loadSmotePreview} disabled={smoteLoading}>
              {smoteLoading ? 'Loading...' : 'Preview SMOTE Balanced Dataset'}
            </button>
          ) : (
            <div className="alert alert-success" style={{ marginTop: '16px' }}>
              ✅ Good Class Balance - Dataset has good distribution across all grades.
            </div>
          )}
        </div>
      )}

      {/* SMOTE Preview */}
      {showSmotePreview && smoteData && (
        <div className="card smote-preview">
          <div className="card-title">
            <FaBalanceScale />
            <span>SMOTE Balanced Dataset Preview</span>
          </div>
          <p style={{ fontSize: '13px', color: '#8b949e', marginBottom: '16px' }}>
            After applying SMOTE, the dataset is balanced with equal samples per class.
            Synthetic samples added: <strong>{smoteData.synthetic_added}</strong>
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Grade</th>
                  <th>Original Samples</th>
                  <th>Balanced Samples</th>
                  <th>Distribution</th>
                </tr>
              </thead>
              <tbody>
                {Object.keys(smoteData.balanced_distribution || {}).map(grade => {
                  const originalCount = smoteData.original_distribution?.[grade] || 0;
                  const balancedCount = smoteData.balanced_distribution[grade];
                  const totalBalanced = Object.values(smoteData.balanced_distribution).reduce((a, b) => a + b, 0);
                  const percent = (balancedCount / totalBalanced * 100).toFixed(1);
                  const isSynthetic = originalCount < balancedCount;

                  return (
                    <tr key={grade} style={isSynthetic ? { background: 'rgba(210, 153, 34, 0.05)' } : {}}>
                      <td><strong>Grade {grade}</strong></td>
                      <td>{originalCount} samples</td>
                      <td>
                        <span style={{ color: getGradeColor(grade), fontWeight: 'bold' }}>
                          {balancedCount} samples
                        </span>
                        {isSynthetic && (
                          <span style={{ marginLeft: '8px', fontSize: '11px', color: '#d29922' }}>
                            (+{balancedCount - originalCount} synthetic)
                          </span>
                        )}
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                          <div style={{ width: '120px', height: '20px', background: '#161b22', borderRadius: '4px', overflow: 'hidden' }}>
                            <div
                              style={{
                                width: `${percent}%`,
                                height: '100%',
                                background: getGradeColor(grade),
                                borderRadius: '4px'
                              }}
                            />
                          </div>
                          <span style={{ fontSize: '12px', fontWeight: '500' }}>{percent}%</span>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <br></br><br></br>
          <button className="btn-secondary" style={{ background: 'linear-gradient(135deg, var(--accent), #a371f7)' }} onClick={() => setShowSmotePreview(false)}>
            <FaEyeSlash /> Hide Preview
          </button>
        </div>
      )}

      {/* Feature Space Visualization */}
      {hasData && (
        <div className="card fade-in">
          <div className="card-title">
            <FaChartLine />
            <span>Feature Space Visualization</span>
          </div>
          <div className="viz-tabs" style={{ gap: '6px', marginBottom: '12px' }}>
  <button
    className={`viz-tab ${activeViz === '2d' ? 'active' : ''}`}
    onClick={() => setActiveViz('2d')}
    style={{ padding: '4px 12px', fontSize: '12px', gap: '4px' }}
  >
    <FaChartBar size={18} /> 2D: Marks vs Grade
  </button>
  <button
    className={`viz-tab ${activeViz === '3d' ? 'active' : ''}`}
    onClick={() => setActiveViz('3d')}
    style={{ padding: '4px 12px', fontSize: '12px', gap: '4px' }}
  >
    <FaCube size={18} /> 3D: Marks + Grade + Study Hours
  </button>
  <button
    className={`viz-tab ${activeViz === 'pca' ? 'active' : ''}`}
    onClick={() => setActiveViz('pca')}
    style={{ padding: '4px 12px', fontSize: '12px', gap: '4px' }}
  >
    <FaChartLine size={18} /> PCA
  </button>
</div>

{activeViz === '2d' && marks2DImage && (
  <img src={marks2DImage} className="viz-image" style={{ maxHeight: '500px', objectFit: 'contain' }} />
)}
{activeViz === '3d' && features3DImage && (
  <img src={features3DImage} className="viz-image" style={{ maxHeight: '500px', objectFit: 'contain' }} />
)}
{activeViz === 'pca' && pcaImage && (
  <img src={pcaImage} className="viz-image" style={{ maxHeight: '500px', objectFit: 'contain' }} />
)}

<button 
  className="btn-secondary" 
  onClick={loadAllVisualizations} 
  disabled={loadingViz}
  style={{ padding: '6px 14px', fontSize: '12px' }}
>
  <FaSync className={loadingViz ? 'spinner' : ''} size={12} /> 
  {loadingViz ? 'Loading...' : 'Refresh'}
</button>
        </div>
      )}

      {/* Data Preview Table */}
      {hasData && (
        <div className="card fade-in">
          <div className="card-title">
            <FaTable />
            <span>First 10 Rows</span>
          </div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  {headers.map(key => <th key={key}>{key}</th>)}
                </tr>
              </thead>
              <tbody>
                {dataPreview.slice(0, 10).map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ textAlign: 'center', fontWeight: 'bold' }}>{idx + 1}</td>
                    {headers.map(key => {
                      const val = row[key];
                      const isGrade = key === 'grade';
                      return (
                        <td key={key}>
                          {isGrade ? (
                            <span className={`badge badge-${val.toLowerCase()}`}>{val}</span>
                          ) : (
                            typeof val === 'number' ? val.toFixed(2) : (val || '-')
                          )}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Empty State */}
      {!hasData && !loading && !dataLoaded && (
        <div className="card empty-state">
          <FaDatabase size={64} style={{ color: '#484f58', marginBottom: '16px' }} />
          <h3>No Data Loaded</h3>
          <p>Click "Load Synthetic Dataset" to get started</p>
        </div>
      )}
    </div>
  );
}

export default Dataset;