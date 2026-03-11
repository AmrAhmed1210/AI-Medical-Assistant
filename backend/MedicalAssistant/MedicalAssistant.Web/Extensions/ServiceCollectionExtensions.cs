using MedicalAssistant.Domain.Contracts;
using MedicalAssistant.Persistance.Repositories;
using MedicalAssistant.Services.Services;
using MedicalAssistant.Services_Abstraction.Contracts;
using Microsoft.Extensions.DependencyInjection;

namespace MedicalAssistant.Web.Extensions
{
    public static class ServiceCollectionExtensions
    {
        /// <summary>
        /// Registers application services for the Patient and Appointment modules (repositories + services).
        /// </summary>
        public static IServiceCollection AddPatientModule(this IServiceCollection services)
        {
            // Unit of Work + repositories
            services.AddScoped<IUnitOfWork, UnitOfWork>();
            services.AddScoped<IAppointmentRepository, AppointmentRepository>();

            // Services
            services.AddScoped<IPatientService, PatientService>();
            services.AddScoped<IAppointmentService, AppointmentService>();

            return services;
        }
    }
}
