
# ============================================================================
# DASHBOARD STATS ENDPOINT
# ============================================================================
@app.get("/stats/dashboard")
async def get_dashboard_stats(current_user: User = Depends(get_current_user)):
    """
    Get aggregated statistics for the current user's dashboard.
    """
    username = current_user.username
    
    # 1. Total Analyses (Jobs requested by user)
    total_query = """
    MATCH (u:User {username: $username})-[:REQUESTED]->(j:Job {status: 'completed'})
    RETURN count(j) as total
    """
    total_result = neo4j_conn.query(total_query, {"username": username})
    total_analyses = total_result[0]['total'] if total_result else 0
    
    # 2. Today's Analyses
    today_query = """
    MATCH (u:User {username: $username})-[:REQUESTED]->(j:Job {status: 'completed'})
    WHERE j.completed_at >= datetime({date: date()})
    RETURN count(j) as today
    """
    today_result = neo4j_conn.query(today_query, {"username": username})
    today_analyses = today_result[0]['today'] if today_result else 0
    
    # 3. Verdict Distribution (from File nodes owned by User)
    verdict_query = """
    MATCH (u:User {username: $username})-[:OWNS]->(f:File)
    RETURN f.verdict as verdict, count(f) as count
    """
    verdict_results = neo4j_conn.query(verdict_query, {"username": username})
    
    deepfakes = 0
    reals = 0
    inconclusive = 0
    
    for row in verdict_results:
        v = row['verdict'].upper()
        if v == 'FAKE':
            deepfakes += row['count']
        elif v == 'REAL':
            reals += row['count']
        else:
            inconclusive += row['count']

    # 4. Weekly Activity (Last 7 days)
    weekly_query = """
    MATCH (u:User {username: $username})-[:REQUESTED]->(j:Job {status: 'completed'})
    WHERE j.completed_at >= datetime() - duration('P7D')
    RETURN date(j.completed_at) as day, count(j) as count
    ORDER BY day
    """
    weekly_results = neo4j_conn.query(weekly_query, {"username": username})
    
    # Fill in missing days for frontend
    # (Simplified: just returning what DB gives, frontend can map or we improve logic)
    weekly_data = [{"day": str(row['day']), "analyses": row['count']} for row in weekly_results]

    # Calculate Trend (vs previous week - Simplified for now)
    # Just returning 0 if no history
    weekly_trend = 0.0 

    return {
        "totalAnalyses": total_analyses,
        "todayAnalyses": today_analyses,
        "weeklyTrend": weekly_trend,
        "deepfakesDetected": deepfakes,
        "authenticVideos": reals,
        "inconclusiveResults": inconclusive,
        "averageProcessingTime": 15000, # Mocked for now, difficult to aggregate without duration stored
        "weeklyData": weekly_data
    }
