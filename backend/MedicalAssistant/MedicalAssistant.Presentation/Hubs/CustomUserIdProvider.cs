using Microsoft.AspNetCore.SignalR;
using System.Security.Claims;

namespace MedicalAssistant.Presentation.Hubs;

public class CustomUserIdProvider : IUserIdProvider
{
    public string? GetUserId(HubConnectionContext connection)
    {
        // Try to get DoctorId first, then PatientId/UserId
        var doctorId = connection.User?.FindFirst("DoctorId")?.Value;
        if (!string.IsNullOrEmpty(doctorId)) return doctorId;

        return connection.User?.FindFirst("UserId")?.Value;
    }
}
