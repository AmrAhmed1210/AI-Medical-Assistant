using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.SignalR;

namespace MedicalAssistant.Presentation.Hubs;

[Authorize]
public class NotificationHub : Hub
{
    public async Task JoinGroup(string groupName)
    {
        await Groups.AddToGroupAsync(Context.ConnectionId, groupName);
    }

    public async Task SubscribeToDoctorSchedule(int doctorId)
    {
        await Groups.AddToGroupAsync(Context.ConnectionId, $"Schedule_Doctor_{doctorId}");
    }

    public override async Task OnConnectedAsync()
    {
        var role = Context.User?.FindFirst(System.Security.Claims.ClaimTypes.Role)?.Value;
        if (!string.IsNullOrWhiteSpace(role))
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, role);
            await Groups.AddToGroupAsync(Context.ConnectionId, role.Trim().ToLowerInvariant());
        }

        var email = Context.User?.FindFirst(System.Security.Claims.ClaimTypes.Email)?.Value;
        if (!string.IsNullOrWhiteSpace(email))
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, email);
            await Groups.AddToGroupAsync(Context.ConnectionId, email.Trim().ToLowerInvariant());
        }

        var doctorId = Context.User?.FindFirst("DoctorId")?.Value;
        if (int.TryParse(doctorId, out var parsedDoctorId) && parsedDoctorId > 0)
        {
            await Groups.AddToGroupAsync(Context.ConnectionId, $"Doctor_{parsedDoctorId}");
            await Groups.AddToGroupAsync(Context.ConnectionId, $"doctor_{parsedDoctorId}");
        }

        await base.OnConnectedAsync();
    }
}
