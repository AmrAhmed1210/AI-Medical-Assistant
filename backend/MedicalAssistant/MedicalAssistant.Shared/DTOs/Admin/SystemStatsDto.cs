namespace MedicalAssistant.Shared.DTOs.Admin;

public class SystemStatsDto
{
    public int TotalUsers { get; set; }
    public int TotalDoctors { get; set; }
    public int TotalPatients { get; set; }
    public int TotalAppointments { get; set; }
    public int ActiveSessions { get; set; }
}