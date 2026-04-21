using MedicalAssistant.Shared.DTOs.AppointmentsDTOs;

namespace MedicalAssistant.Shared.DTOs.DoctorDTOs;

public class DoctorDashboardDto
{
    public int TodayAppointments { get; set; }
    public int PendingAppointments { get; set; }
    public int TotalPatients { get; set; }
    public int WeekAppointments { get; set; }
    public List<AppointmentDto> TodayAppointmentsList { get; set; } = new();
    public List<ChartDataDto> WeeklySessionsChart { get; set; } = new();
    public List<AIReportDto> RecentReports { get; set; } = new();
}