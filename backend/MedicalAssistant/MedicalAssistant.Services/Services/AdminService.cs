using AutoMapper;
using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Domain.Entities.UserModule;
using MedicalAssistant.Domain.Entities.AppointmentsModule;
using MedicalAssistant.Domain.Entities.AnalysisModule;
using MedicalAssistant.Services_Abstraction.Contracts;
using MedicalAssistant.Shared.DTOs.Admin;
using MedicalAssistant.Shared.DTOs.Common;

namespace MedicalAssistant.Application.Services
{
    public class AdminService : IAdminService
    {
        private readonly IUnitOfWork _unitOfWork;
        private readonly IMapper _mapper;

        public AdminService(IUnitOfWork unitOfWork, IMapper mapper)
        {
            _unitOfWork = unitOfWork;
            _mapper = mapper;
        }

        public async Task<SystemStatsDto> GetSystemStatsAsync()
        {
            var users = await _unitOfWork.Repository<User>().GetAllAsync();
            var appointments = await _unitOfWork.Repository<Appointment>().GetAllAsync();
            var sessions = await _unitOfWork.Repository<AnalysisResult>().GetAllAsync();

            return new SystemStatsDto
            {
                TotalUsers = users.Count(),
                TotalDoctors = users.Count(u => u.Role == "Doctor"),
                TotalPatients = users.Count(u => u.Role == "Patient"),
                TotalAppointments = appointments.Count(),
                TotalSessions = sessions.Select(s => s.SessionId).Distinct().Count(),
                ActiveModels = 0,
                AvgResponseTimeMs = 0,
                HighUrgencyToday = sessions.Count(s => s.UrgencyLevel == "HIGH" && s.CreatedAt.Date == DateTime.UtcNow.Date)
            };
        }

        public async Task<PagedResult<UserManagementDto>> GetUsersAsync(int page, int pageSize, string? search, string? role)
        {
            var allUsers = await _unitOfWork.Repository<User>().GetAllAsync();
            var query = allUsers.Where(u => !u.IsDeleted);

            if (!string.IsNullOrEmpty(role))
                query = query.Where(u => u.Role.Equals(role, StringComparison.OrdinalIgnoreCase));

            if (!string.IsNullOrEmpty(search))
                query = query.Where(u => u.FullName.Contains(search, StringComparison.OrdinalIgnoreCase) ||
                                         u.Email.Contains(search, StringComparison.OrdinalIgnoreCase));

            var total = query.Count();
            var items = query.Skip((page - 1) * pageSize).Take(pageSize).ToList();

            return new PagedResult<UserManagementDto>
            {
                Items = _mapper.Map<IEnumerable<UserManagementDto>>(items),
                Total = total
            };
        }

        public async Task<UserManagementDto> CreateUserAsync(CreateUserRequest request)
        {
            var user = _mapper.Map<User>(request);
            await _unitOfWork.Repository<User>().AddAsync(user);
            await _unitOfWork.SaveChangesAsync();
            return _mapper.Map<UserManagementDto>(user);
        }

        public async Task<bool> ToggleUserStatusAsync(int userId)
        {
            var user = await _unitOfWork.Repository<User>().GetByIdAsync(userId);
            if (user == null) return false;

            user.IsActive = !user.IsActive;
            user.UpdatedAt = DateTime.UtcNow;
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public async Task<bool> DeleteUserAsync(int id)
        {
            var user = await _unitOfWork.Repository<User>().GetByIdAsync(id);
            if (user == null) return false;

            user.IsDeleted = true;
            user.UpdatedAt = DateTime.UtcNow;
            await _unitOfWork.SaveChangesAsync();
            return true;
        }

        public Task<IEnumerable<ModelVersionDto>> ListModelVersionsAsync()
        {
            return Task.FromResult(Enumerable.Empty<ModelVersionDto>());
        }

        public Task ReloadAiModelAsync(string agentName)
        {
            return Task.CompletedTask;
        }
    }
}