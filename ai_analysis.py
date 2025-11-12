"""
Gemini AI Integration for Inventory Analysis
Provides AI-powered insights, summaries, and report generation
"""

import json
from typing import Dict, Any, Optional
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


def get_gemini_analysis(
    api_key: str,
    analysis_type: str,
    data_summary: Dict[str, Any],
    sample_data: Dict[str, Any]
) -> str:
    """
    Get AI analysis from Gemini API
    
    Args:
        api_key: Gemini API key
        analysis_type: Type of analysis requested
        data_summary: Summary of inventory data
        sample_data: Sample data for context
    
    Returns:
        Formatted analysis text
    """
    try:
        import google.generativeai as genai
        
        # Configure Gemini - use free tier model
        genai.configure(api_key=api_key)
        # Use gemini-1.5-flash for free tier (faster and free)
        # Fallback to gemini-1.5-pro if flash is not available
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
        except:
            try:
                model = genai.GenerativeModel('gemini-1.5-pro')
            except:
                model = genai.GenerativeModel('gemini-pro')
        
        # Create prompt based on analysis type
        prompt = _create_analysis_prompt(analysis_type, data_summary, sample_data)
        
        # Generate response
        response = model.generate_content(prompt)
        
        return response.text
        
    except ImportError:
        # Fallback if google-generativeai is not installed
        return _generate_fallback_analysis(analysis_type, data_summary)
    except Exception as e:
        return f"Error generating AI analysis: {str(e)}\n\nPlease check your API key and ensure google-generativeai package is installed."


def _create_analysis_prompt(
    analysis_type: str,
    data_summary: Dict[str, Any],
    sample_data: Dict[str, Any]
) -> str:
    """Create prompt for Gemini based on analysis type"""
    
    base_context = f"""
    You are an expert inventory management analyst. Analyze the following inventory data and provide insights.
    
    Data Summary:
    - Total Items: {data_summary.get('total_items', 'N/A')}
    - Columns: {', '.join(data_summary.get('columns', []))}
    - Summary Statistics: {json.dumps(data_summary.get('summary_stats', {}), indent=2)}
    """
    
    if analysis_type == "📊 Overall Inventory Summary":
        prompt = f"""
        {base_context}
        
        Provide a comprehensive summary of the inventory data including:
        1. Overall inventory health assessment
        2. Key statistics and patterns
        3. Potential areas of concern
        4. General recommendations
        
        Format the response in clear sections with bullet points where appropriate.
        """
    
    elif analysis_type == "📈 EOQ Analysis Interpretation":
        prompt = f"""
        {base_context}
        
        Interpret the EOQ (Economic Order Quantity) analysis results. Explain:
        1. What the EOQ values indicate about inventory management
        2. Cost optimization opportunities
        3. Recommendations for ordering strategies
        4. Potential improvements
        
        Be specific and actionable in your recommendations.
        """
    
    elif analysis_type == "🔤 ABC Analysis Insights":
        prompt = f"""
        {base_context}
        
        Provide insights on the ABC analysis results:
        1. Interpretation of category distribution
        2. Strategic recommendations for each category (A, B, C)
        3. Resource allocation suggestions
        4. Priority management strategies
        
        Focus on actionable insights for inventory optimization.
        """
    
    elif analysis_type == "💰 Cost Optimization Recommendations":
        prompt = f"""
        {base_context}
        
        Provide cost optimization recommendations:
        1. Identify high-cost items and areas
        2. Suggest specific cost reduction strategies
        3. Optimize ordering and holding costs
        4. Calculate potential savings
        
        Be quantitative where possible.
        """
    
    else:  # Comprehensive Report
        prompt = f"""
        {base_context}
        
        Create a comprehensive inventory management report covering:
        1. Executive Summary
        2. Current Inventory Status
        3. EOQ Analysis and Recommendations
        4. ABC Analysis Insights
        5. Forecasting Accuracy and Recommendations
        6. Cost Optimization Opportunities
        7. Action Items and Next Steps
        
        Make it professional and suitable for management review.
        """
    
    return prompt


def _generate_fallback_analysis(analysis_type: str, data_summary: Dict[str, Any]) -> str:
    """Generate basic analysis without AI if Gemini is not available"""
    
    analysis = f"""
    # {analysis_type}
    
    ## Data Overview
    - Total Items: {data_summary.get('total_items', 'N/A')}
    - Data Columns: {', '.join(data_summary.get('columns', []))}
    
    ## Key Insights
    
    ### Inventory Status
    The inventory dataset contains {data_summary.get('total_items', 0)} items with various characteristics.
    
    ### Recommendations
    1. Review EOQ calculations for cost optimization
    2. Implement ABC analysis for prioritized management
    3. Use forecasting models to improve demand prediction
    4. Monitor key performance metrics regularly
    
    **Note:** For detailed AI-powered analysis, please ensure the Gemini API key is configured and the google-generativeai package is installed.
    """
    
    return analysis


def generate_pdf_report(
    api_key: str,
    title: str,
    analysis_text: str,
    analysis_type: str,
    data_summary: Dict[str, Any]
) -> bytes:
    """
    Generate PDF report using ReportLab
    
    Args:
        api_key: Gemini API key (for potential future enhancements)
        title: Report title
        analysis_text: AI-generated analysis text
        analysis_type: Type of analysis
        data_summary: Summary of data
    
    Returns:
        PDF file as bytes
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    story = []
    
    # Define styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.HexColor('#2c3e50'),
        spaceAfter=12,
        spaceBefore=12
    )
    
    # Title
    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 0.2*inch))
    
    # Report metadata
    from datetime import datetime
    metadata = f"""
    <b>Report Type:</b> {analysis_type}<br/>
    <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
    <b>Total Items Analyzed:</b> {data_summary.get('total_items', 'N/A')}
    """
    story.append(Paragraph(metadata, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Analysis content
    # Split analysis text into paragraphs
    paragraphs = analysis_text.split('\n\n')
    
    for para in paragraphs:
        if para.strip():
            # Check if it's a heading
            if para.startswith('#') or para.startswith('##'):
                # Remove markdown heading symbols
                heading_text = para.lstrip('#').strip()
                story.append(Paragraph(heading_text, heading_style))
            else:
                # Regular paragraph
                # Convert markdown to HTML-like format
                para_html = para.replace('\n', '<br/>')
                story.append(Paragraph(para_html, styles['Normal']))
                story.append(Spacer(1, 0.1*inch))
    
    # Add summary table
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Data Summary", heading_style))
    
    summary_data = [
        ['Metric', 'Value'],
        ['Total Items', str(data_summary.get('total_items', 'N/A'))],
        ['Columns', ', '.join(data_summary.get('columns', []))[:50] + '...' if len(data_summary.get('columns', [])) > 50 else ', '.join(data_summary.get('columns', []))]
    ]
    
    # Add summary statistics if available
    if data_summary.get('summary_stats'):
        for key, value in list(data_summary['summary_stats'].items())[:5]:
            if isinstance(value, dict):
                summary_data.append([key, str(list(value.values())[0])[:30]])
    
    table = Table(summary_data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer.getvalue()

