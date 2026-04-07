public class ModelVersionDto
{
    public int Id { get; set; }
    public string AgentName { get; set; } = string.Empty;
    public string Version { get; set; } = string.Empty;
    public bool IsActive { get; set; }
    public DateTime? LoadedAt { get; set; }
}
